import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import math
import os
from functools import partial
from tqdm import tqdm
from evaluate_utils import load_jsonl, get_duration_by_youtube_id

class JIREvaluator:
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 SENTENCE_DURATION: int = 5,
                 HUMAN_ANNOTATION_LIKELIHOOD_SCORE: int = 9):
        """
        Initialize the JIR evaluator.
        
        Args:
            similarity_threshold: Threshold for semantic similarity (0 to 1)
        """
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.SENTENCE_DURATION = SENTENCE_DURATION
        self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE = HUMAN_ANNOTATION_LIKELIHOOD_SCORE
        
    def _check_time_overlap(self, q1_start: float, q1_end: float, 
                           q2_start: float, q2_end: float, fuzzy_sentence_interval=0) -> bool:
        """Check if two time intervals overlap."""
        # Create local copies for effective start and end times
        effective_q1_start = q1_start
        effective_q1_end = q1_end
        
        # Apply fuzzy interval to start time (extend earlier)
        effective_q1_start = q1_start - (fuzzy_sentence_interval * self.SENTENCE_DURATION)
        
        # Handle the case where effective_q1_end is None or equal to q1_start
        if effective_q1_end is None or q1_start == effective_q1_end:
            # Extend effective_q1_end to accommodate two sentences worth of time
            effective_q1_end = q1_start + (2 * self.SENTENCE_DURATION)
        
        # Apply fuzzy interval to end time (extend later)
        effective_q1_end = effective_q1_end + (fuzzy_sentence_interval * self.SENTENCE_DURATION)

        effective_q2_end = q2_end if q2_end else q2_start + (2 * self.SENTENCE_DURATION)
        
        # Check for overlap using the potentially modified times
        return max(effective_q1_start, q2_start) <= min(effective_q1_end, effective_q2_end)
    
    def _compute_similarity(self, query1: str, query2: str) -> float:
        """Compute semantic similarity between two queries."""
        emb1 = self.model.encode(query1, convert_to_tensor=True)
        emb2 = self.model.encode(query2, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), 
                                                          emb2.unsqueeze(0)).item()
        return similarity
        
    def find_matches(self, 
                    query_to_match: Dict[str, Any], 
                    candidate_queries: List[Dict[str, Any]],
                    fuzzy_sentence_interval: int=3) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find matches for a query from a list of candidate queries.
        
        Args:
            query_to_match: Query to be matched with its attributes
            candidate_queries: List of candidate queries to match against
            
        Returns:
            List of tuples containing matched queries and their similarity scores
        """
        matches = []
        
        for candidate in candidate_queries:
            # Check time overlap
            time_overlaps = self._check_time_overlap(
                query_to_match['start_time'], query_to_match['end_time'],
                candidate['start_time'], candidate['end_time'],
                fuzzy_sentence_interval
            )
            
            if time_overlaps:
                # Compute semantic similarity
                similarity = self._compute_similarity(
                    query_to_match['question'], candidate['question']
                )
                
                if similarity >= self.similarity_threshold:
                    matches.append((candidate, similarity))
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def batch_find_matches(self, 
                          queries_to_match: List[Dict[str, Any]], 
                          candidate_queries: List[Dict[str, Any]],
                          fuzzy_sentence_interval: int=3) -> Dict[int, List[Tuple[Dict[str, Any], float]]]:
        """
        Find matches for multiple queries from a list of candidate queries.
        
        Args:
            queries_to_match: List of queries to be matched
            candidate_queries: List of candidate queries to match against
            
        Returns:
            Dictionary mapping query index to its matches
        """
        results = {}
        
        for i, query in enumerate(queries_to_match):
            matches = self.find_matches(query, candidate_queries, fuzzy_sentence_interval)
            results[i] = matches
            
        return results

    def compute_precision(self, 
                      candidate_queries: List[Dict[str, Any]], 
                      results: Dict[int, List[Tuple[Dict[str, Any], float]]]) -> Dict[str, float]:
        """
        Compute precision: the percentage of candidate queries that find a match.
        
        Args:
            candidate_queries: List of candidate queries to match against
            results: Dictionary mapping query index to its matches
            
        Returns:
            Dictionary containing precision metric and detail counts
        """
        # Create a set of all unique candidate queries that were matched
        matched_candidates = set()
        for matches in results.values():
            for candidate, _ in matches:
                # Use a unique identifier for the candidate
                candidate_id = candidate["question"]
                matched_candidates.add(candidate_id)
        
        # Calculate metrics
        total_candidates = len(candidate_queries)
        matched_candidates_count = len(matched_candidates)
        
        # Calculate precision
        precision = matched_candidates_count / total_candidates if total_candidates > 0 else 0
        
        return {
            'precision': precision,
            'matched_candidates': matched_candidates_count,
            'total_candidates': total_candidates
        }

    def compute_recall(self, 
                    queries_to_match: List[Dict[str, Any]], 
                    results: Dict[int, List[Tuple[Dict[str, Any], float]]]) -> Dict[str, float]:
        """
        Compute recall: the percentage of queries to be matched that find match(es).
        
        Args:
            queries_to_match: List of queries that need to be matched
            results: Dictionary mapping query index to its matches
            
        Returns:
            Dictionary containing recall metric and detail counts
        """
        # Count how many queries_to_match found at least one match
        matched_queries_count = sum(1 for matches in results.values() if len(matches) > 0)
        
        # Calculate metrics
        total_queries_to_match = len(queries_to_match)
        
        # Calculate recall
        recall = matched_queries_count / total_queries_to_match if total_queries_to_match > 0 else 0
        
        return {
            'recall': recall,
            'matched_queries': matched_queries_count,
            'total_queries': total_queries_to_match
        }
    
    def compute_ndcg(self, relevance_scores: List[int], ideal_scores: List[int], k: int = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (nDCG) for a list of relevance scores.
        
        Args:
            relevance_scores: List of relevance scores (higher is better)
            k: Number of items to consider (if None, consider all)
            
        Returns:
            nDCG score between 0 and 1
        """
        if not relevance_scores:
            return 0.0
        
        if k is not None:
            if k < len(relevance_scores):
                relevance_scores = relevance_scores[:k]
            elif k > len(relevance_scores):
                relevance_scores.extend([0] * (k - len(relevance_scores)))
            if k < len(ideal_scores):
                ideal_scores = ideal_scores[:k]
            elif k > len(ideal_scores):
                ideal_scores.extend([0] * (k - len(ideal_scores)))
        
        # Calculate DCG
        dcgs = [relevance_scores[0]]
        dcg = relevance_scores[0]
        for i, score in enumerate(relevance_scores[1:], start=2):
            dcg += score / np.log2(i)
            dcgs.append(dcg)
        
        # Calculate ideal DCG (sorted relevance scores)
        idcgs = [ideal_scores[0]]
        idcg = ideal_scores[0]
        for i, score in enumerate(ideal_scores[1:], start=2):
            idcg += score / np.log2(i)
            idcgs.append(idcg)
                
        # Return nDCG
        return sum(dcgs) / sum(idcgs) / len(idcgs)

    def evaluate_relevance(
        self,
        results: Dict[int, List[Tuple[Dict[str, Any], float]]],
        queries_to_match: List[Dict[str, Any]],
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate information retrieval results using nDCG.
        
        Args:
            results: Dictionary mapping query index to its matches
            queries_to_match: Original queries to be matched
            k: Number of top results to consider for nDCG
            
        Returns:
            Dictionary with evaluation metrics
        """
        query_scores = {}
        total_likelihood = sum(query.get('likelihood_score', self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE) for query in queries_to_match)
        matched_total_likelihood = sum(query.get('likelihood_score', self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE) for i, query in enumerate(queries_to_match) if len(results[i])>0)

        weighted_sum = 0.0
        matched_weighted_sum = 0.0
        
        for idx, matches in results.items():
            query = queries_to_match[idx]
            
            # Get document relevance scores for this query
            doc_relevance = query["references"].get('document_relevance_score', {})
            
            # If no matches found, score is 0
            if not matches:
                query_scores[idx] = 0.0
                continue
                
            # Extract relevance scores for the matched documents
            relevance_scores = []
            matches = sorted(matches, key=lambda item: item[1], reverse=True)
            docs = [s for m in matches for s in m[0]["references"]]
            relevance_scores = [doc_relevance[d] if d in doc_relevance else 0 for d in docs]

            ideal_scores = sorted([s for s in doc_relevance.values()], reverse=True)
            
            # Compute nDCG for this query
            ndcg_score = self.compute_ndcg(relevance_scores, ideal_scores, k)
            query_scores[idx] = ndcg_score
            
            # For weighted aggregate
            query_weight = query.get('likelihood_score', self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE) / total_likelihood
            weighted_sum += ndcg_score * query_weight
            matched_query_weight = query.get('likelihood_score', self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE) / matched_total_likelihood
            matched_weighted_sum += ndcg_score * matched_query_weight
        
        # Prepare results
        evaluation = {
            'query_scores': query_scores,
            'average_ndcg': np.mean(list(query_scores.values())) if query_scores else 0.0,
            'weighted_ndcg': weighted_sum,
            'matched_weighted_ndcg': matched_weighted_sum,
            'num_queries': len(query_scores),
            'num_queries_with_relevant_references': sum(1 for score in query_scores.values() if score > 0),
            'k': k
        }
        
        return evaluation

    def compute_time_match_score(self, t1: float, t2: float, delta: float = 2.0) -> float:
        """
        Compute Gaussian-like kernel score for two time points.
        
        Args:
            t1: First time point
            t2: Second time point
            delta: Scaling parameter for the Gaussian kernel
            
        Returns:
            Score between 0 and 1, with 1 being perfect match
        """
        return math.exp(-((t1 - t2) ** 2) / (2*(delta ** 2)))

    def evaluate_timeliness(
        self, 
        queries_to_match: List[Dict[str, Any]], 
        results: Dict[int, List[Tuple[Dict[str, Any], float]]],
        fuzzy_sentence_interval: int = 2,
        balancing_coeffient: float = 0.9
    ) -> Dict[str, Any]:
        """
        Evaluate time matching for all queries using a Gaussian kernel.
        
        Args:
            queries_to_match: Original queries to be matched
            results: Dictionary mapping query index to its matches
            delta: Scaling parameter for the Gaussian kernel
            
        Returns:
            Dictionary with time matching evaluation metrics
        """
        query_time_scores = {}
        total_likelihood = sum(query.get('likelihood_score', self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE) for query in queries_to_match)
        matched_total_likelihood = sum(query.get('likelihood_score', self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE) for i, query in enumerate(queries_to_match) if len(results[i])>0)
        
        # Track aggregated metrics
        weighted_start_sum = 0.0
        weighted_end_sum = 0.0
        weighted_avg_sum = 0.0
        matched_weighted_start_sum = 0.0
        matched_weighted_end_sum = 0.0
        matched_weighted_avg_sum = 0.0
        
        all_start_scores = []
        all_end_scores = []
        all_avg_scores = []
        
        for idx, matches in results.items():
            query = queries_to_match[idx]
            query_weight = query.get('likelihood_score', self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE) / total_likelihood
            matched_query_weight = query.get('likelihood_score', self.HUMAN_ANNOTATION_LIKELIHOOD_SCORE) / matched_total_likelihood
            
            # If no matches found, score is 0
            if not matches:
                time_scores = {
                    'start_time_match': 0.0,
                    'end_time_match': 0.0,
                    'avg_time_match': 0.0
                }
            else:
                # Extract query times
                query_start = query.get('start_time', 0.0)
                query_end = query.get('end_time', query_start)
                if query_end is None or query_end == query_start:
                    query_end = query_start + 2 * self.SENTENCE_DURATION
                
                # Compute match scores for all candidates and take the best one
                best_start_match = np.inf
                best_end_match = 0.0
                
                for match, _ in matches:
                    match_start = match.get('start_time', 0.0)
                    match_end = match.get('end_time', 0.0)    
                    
                    # Update best scores
                    best_start_match = min(best_start_match, match_start)
                    best_end_match = max(best_end_match, match_end)
                
                # Compute time match scores
                start_score = self.compute_time_match_score(query_start, best_start_match, fuzzy_sentence_interval * self.SENTENCE_DURATION)
                end_score = self.compute_time_match_score(query_end, best_end_match, fuzzy_sentence_interval * self.SENTENCE_DURATION)
                # Calculate average time match score
                avg_time_match = balancing_coeffient * start_score + (1 - balancing_coeffient) * end_score
                
                time_scores = {
                    'start_time_match': start_score,
                    'end_time_match': end_score,
                    'avg_time_match': avg_time_match
                }
            
            query_time_scores[idx] = time_scores
            
            # Update aggregated metrics
            weighted_start_sum += time_scores['start_time_match'] * query_weight
            weighted_end_sum += time_scores['end_time_match'] * query_weight
            weighted_avg_sum += time_scores['avg_time_match'] * query_weight
            matched_weighted_start_sum += time_scores['start_time_match'] * matched_query_weight
            matched_weighted_end_sum += time_scores['end_time_match'] * matched_query_weight
            matched_weighted_avg_sum += time_scores['avg_time_match'] * matched_query_weight
            
            all_start_scores.append(time_scores['start_time_match'])
            all_end_scores.append(time_scores['end_time_match'])
            all_avg_scores.append(time_scores['avg_time_match'])
        
        def compute_time_error(timeliness_score, delta):
            return math.sqrt(-math.log(timeliness_score)*(2*(delta ** 2)))
        
        customed_compute_time_error = partial(compute_time_error, delta=fuzzy_sentence_interval*self.SENTENCE_DURATION)
        # Calculate aggregated metrics
        evaluation = {
            'query_time_scores': query_time_scores,
            'average_start_match': np.mean(all_start_scores) if all_start_scores else 0.0,
            'average_end_match': np.mean(all_end_scores) if all_end_scores else 0.0,
            'average_time_match': np.mean(all_avg_scores) if all_avg_scores else 0.0,
            'weighted_start_match': weighted_start_sum,
            'weighted_end_match': weighted_end_sum,
            'weighted_time_match': weighted_avg_sum,
            'matched_weighted_start_match': matched_weighted_start_sum,
            'matched_weighted_end_match': matched_weighted_end_sum,
            'matched_weighted_time_match': matched_weighted_avg_sum,
            'weighted_start_error': customed_compute_time_error(weighted_start_sum),
            'weighted_end_error': customed_compute_time_error(weighted_end_sum),
            'weighted_time_error': customed_compute_time_error(weighted_avg_sum),
            'matched_weighted_start_error': customed_compute_time_error(matched_weighted_start_sum),
            'matched_weighted_end_error': customed_compute_time_error(matched_weighted_end_sum),
            'matched_weighted_time_error': customed_compute_time_error(matched_weighted_avg_sum),
            'num_queries': len(query_time_scores),
            'num_queries_with_matches': sum(1 for scores in query_time_scores.values() 
                                        if scores['avg_time_match'] > 0),
            'delta': fuzzy_sentence_interval * self.SENTENCE_DURATION,
            'balancing_coeffient': balancing_coeffient
        }
        
        return evaluation

    def text_interactive_visualization(self, 
                                    queries_to_match: List[Dict[str, Any]], 
                                    candidate_queries: List[Dict[str, Any]],
                                    match_results: Dict[int, List[Tuple[Dict[str, Any], float]]] = None,
                                    fuzzy_sentence_interval: int = 3) -> None:
        """
        Create a text-based interactive visualization where you can type query IDs
        to see all candidates with time overlap.
        
        Args:
            queries_to_match: List of queries to be matched
            candidate_queries: List of candidate queries to match against
            match_results: Optional dictionary of matching results from batch_find_matches
            fuzzy_sentence_interval: Number of sentences to extend for fuzzy time matching
        """
        import time
        
        # If match_results not provided, generate them
        if match_results is None:
            match_results = self.batch_find_matches(queries_to_match, candidate_queries)
        
        print("\n===== TEXT-BASED INTERACTIVE QUERY VISUALIZATION =====\n")
        print(f"Total queries: {len(queries_to_match)}")
        print(f"Total candidates: {len(candidate_queries)}")
        print(f"Fuzzy time interval: {fuzzy_sentence_interval} sentences\n")
        
        # First, display a summary of all queries
        print("AVAILABLE QUERIES:")
        print("-----------------")
        for i, query in enumerate(queries_to_match):
            start = query.get('start_time', 0)
            end = query.get('end_time', start)
            if end is None or end == start:
                end = start + 2 * self.SENTENCE_DURATION
            
            likelihood_score = sum(x["score"] for x in query["likelihood_scores"]) / len(query["likelihood_scores"]) if query.get("likelihood_scores", None) else ("human" if query.get("data_type") == "human" else None)
                
            question = query.get('question', '')
            question = question[:100] + "..." if len(question) > 100 else question
            
            # Check if this query has any matches
            has_matches = i in match_results and len(match_results[i]) > 0
            match_indicator = "✓" if has_matches else "✗"
            def color_score(score):
                if isinstance(score, str):
                    return f"\033[94m{score}\033[0m"
                elif score >= 9:
                    return f"\033[92m{score}\033[0m"
                elif score >= 8:
                    return f"\033[93m{score}\033[0m"
                else:
                    return f"\033[91m{score}\033[0m"

            def color_match(match):
                return f"\033[95m{match}\033[0m" if match == "✗" else f"\033[96m{match}\033[0m"
            
            if likelihood_score:
                print(f"{i}: [{start:.1f}s - {end:.1f}s] ({color_score(likelihood_score)}) {color_match(match_indicator)} {question}")
            else:
                print(f"{i}: [{start:.1f}s - {end:.1f}s] {color_match(match_indicator)} {question}")


        
        # Main interaction loop
        while True:
            print("\nEnter a query ID to see overlapping candidates (or 'q' to quit): ", end="")
            user_input = input().strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Exiting visualization...")
                break
            
            try:
                query_idx = int(user_input)
                if query_idx < 0 or query_idx >= len(queries_to_match):
                    print(f"Error: Query ID must be between 0 and {len(queries_to_match)-1}")
                    continue
                    
                # Show details for the selected query
                query = queries_to_match[query_idx]
                query_start = query.get('start_time', 0)
                query_end = query.get('end_time', query_start)
                if query_end is None or query_end == query_start:
                    query_end = query_start + 2 * self.SENTENCE_DURATION
                    
                # Calculate effective query time range with fuzzy interval
                effective_start = query_start - (fuzzy_sentence_interval * self.SENTENCE_DURATION)
                effective_end = query_end + (fuzzy_sentence_interval * self.SENTENCE_DURATION)
                
                print("\n" + "="*80)
                print(f"QUERY {query_idx} DETAILS:")
                print(f"Time: [{query_start:.1f}s - {query_end:.1f}s] (Effective with fuzzy: [{effective_start:.1f}s - {effective_end:.1f}s])")
                print(f"Question: {query.get('question', '')}")
                
                # Get matches for this query
                semantic_matches = match_results.get(query_idx, [])
                matched_candidates = [c for c, _ in semantic_matches]
                
                # Find all candidates with time overlap
                overlapping_candidates = []
                for j, candidate in enumerate(candidate_queries):
                    if 'start_time' in candidate:
                        cand_start = candidate.get('start_time', 0)
                        cand_end = candidate.get('end_time', cand_start)
                        if cand_end is None or cand_end == cand_start:
                            cand_end = cand_start + 2 * self.SENTENCE_DURATION
                            
                        # Check for time overlap
                        time_overlaps = self._check_time_overlap(
                            effective_start, effective_end,
                            cand_start, cand_end
                        )
                        
                        if time_overlaps:
                            overlapping_candidates.append((j, candidate, candidate in matched_candidates))
                
                # Display overlapping candidates
                print("\nCANDIDATES WITH TIME OVERLAP:")
                print(f"Found {len(overlapping_candidates)} candidates with time overlap")
                print("-" * 80)
                
                if not overlapping_candidates:
                    print("No candidates with time overlap found.")
                else:
                    for j, candidate, is_match in overlapping_candidates:
                        cand_start = candidate.get('start_time', 0)
                        cand_end = candidate.get('end_time', cand_start)
                        if cand_end is None or cand_end == cand_start:
                            cand_end = cand_start + 2 * self.SENTENCE_DURATION
                        
                        # Get similarity score if it's a match
                        sim_score = 0
                        for c, score in semantic_matches:
                            if c == candidate:
                                sim_score = score
                                break
                        
                        # Determine match status indicators
                        if is_match:
                            status = f"✓ MATCH (sim={sim_score:.2f})"
                        else:
                            sim_score = self._compute_similarity(query.get('question', ''), candidate.get('question', ''))
                            status = f"✗ TIME OVERLAP ONLY (sim={sim_score:.2f})"
                        
                        print(f"C{j}: [{cand_start:.1f}s - {cand_end:.1f}s] {status}")
                        print(f"    Question: {candidate.get('question', '')}")
                        print("-" * 80)
                                
            except ValueError:
                print("Error: Please enter a valid query ID number")
                
        print("\nVisualization session ended.")
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_input_dir', type=str, required=True)
    parser.add_argument('--evaluation_output_dir', type=str, required=True)
    parser.add_argument('--evaluation_ground_truth_dir', type=str, required=True)

    args = parser.parse_args()

    EVALUATION_INPUT_DIR = args.evaluation_input_dir
    EVALUATION_OUTPUT_DIR = args.evaluation_output_dir
    EVALUATION_GROUND_TRUTH_DIR = args.evaluation_ground_truth_dir
    for model_id in ["claude-3-7", "DeepSeek-V3-0324", "gemini", "gpt-4o"]:
        os.makedirs(f"{EVALUATION_OUTPUT_DIR}_{model_id}", exist_ok=True)
        sum_dict, has_matches_dict = {}, {}
        for filename in tqdm(os.listdir(f"{EVALUATION_INPUT_DIR}_{model_id}"), model_id):
            youtube_id = filename[:-len(".json")]
            duration = get_duration_by_youtube_id(youtube_id)
            evaluation_output_filepath = os.path.join(f"{EVALUATION_OUTPUT_DIR}_{model_id}/{youtube_id}.json")
            if os.path.exists(evaluation_output_filepath):
                continue
            if not os.path.exists(f"{EVALUATION_GROUND_TRUTH_DIR}/{youtube_id}/jir_references_relevance_score.jsonl"):
                continue
            queries_to_match = load_jsonl(f"{EVALUATION_GROUND_TRUTH_DIR}/{youtube_id}/jir_references_relevance_score.jsonl")
            candidate_queries = json.load(open(os.path.join(f"{EVALUATION_INPUT_DIR}_{model_id}", filename), "r"))["needs"]
            similarity_threshold = 0.55
            fuzzy_sentence_interval = 1

            evaluator = JIREvaluator(similarity_threshold=similarity_threshold)
            results = evaluator.batch_find_matches(queries_to_match, candidate_queries, fuzzy_sentence_interval)

            recall = evaluator.compute_recall(queries_to_match, results)
            precision = evaluator.compute_precision(candidate_queries, results)
            relevance = evaluator.evaluate_relevance(results, queries_to_match)
            timeliness = evaluator.evaluate_timeliness(queries_to_match, results)

            evaluation_outcomes = {
                "duration": duration,
                "recall": recall,
                "precision": precision,
                "relevance": relevance,
                "timeliness": timeliness
            }

            json.dump(evaluation_outcomes, open(evaluation_output_filepath, "w"))