#!/bin/bash

# Configuration
NEED_MODEL_IDS=("gpt-4o" "DeepSeek-V3-0324")
JUDGE_END_TIME_MODEL_IDS=("gpt-4o" "DeepSeek-V3-0324")
JUDGE_SCORE_MODEL_IDS=("gpt-4o" "DeepSeek-V3-0324")
REPLICATION_FACTOR=1
NEED_CHUNK_SIZE=5
FOLDER_NAME="sSL3KMwZ7vI"
PROJECT_FOLDER="output/$FOLDER_NAME"
NARRATIVE_PATH="formatted/$FOLDER_NAME/formatted_narrative.json"
TRANSCRIPT_PATH="formatted/$FOLDER_NAME/formatted_transcript.json"
mkdir -p "$PROJECT_FOLDER"

# # # Step 1: Parallelized per model and replication for user_information_need_simulation.video_agent and user_information_need_simulation.info_need_agent
pids=()
need_index=1
for model_id in "${NEED_MODEL_IDS[@]}"; do
    for ((rep=1; rep<=REPLICATION_FACTOR; rep++)); do
        (
            NEED_FOLDER="$PROJECT_FOLDER/need_$need_index"
            mkdir -p "$NEED_FOLDER"
            echo "$model_id" > "$NEED_FOLDER/model.txt"

            # Parallelize user_information_need_simulation.video_agent and user_information_need_simulation.info_need_agent for the same need_x
            echo "[need_$need_index | model: $model_id] Running user_information_need_simulation.video_agent"
            python3 -m user_information_need_simulation.video_agent \
                --output_path "$NEED_FOLDER/video_need.json" \
                --video_narrative "$NARRATIVE_PATH" \
                --chunk_size $NEED_CHUNK_SIZE \
                --model_id "$model_id" &  # Run in the background

            echo "[need_$need_index | model: $model_id] Running user_information_need_simulation.info_need_agent"
            python3 -m user_information_need_simulation.info_need_agent \
                --output_path "$NEED_FOLDER/info_need.json" \
                --transcript_path "$TRANSCRIPT_PATH" \
                --chunk_size $NEED_CHUNK_SIZE \
                --model_id "$model_id" &  # Run in the background

            wait  # Wait for both background jobs to finish
        ) &
        pids+=($!)
        ((need_index++))
    done
done

# # # # Wait for all backgrounded subshells
for pid in "${pids[@]}"; do
    wait "$pid"
done

# # # Step 2: Parallelize judging end time stamp per need
pids=()
judge_index=1
for need_folder in "$PROJECT_FOLDER"/need_*; do
    (
        model_id=$(cat "$need_folder/model.txt")
        # for model_id in "${JUDGE_END_TIME_MODEL_IDS[@]}"; do
        JUDGE_FOLDER="$need_folder/judge_time_$judge_index"
        mkdir -p "$JUDGE_FOLDER"
        echo "[$need_folder| model: $model_id] Running user_information_need_simulation.llm_judge on info_need"
        python3 -m user_information_need_simulation.llm_judge \
            --transcript_file "$TRANSCRIPT_PATH" \
            --narrative_file "$NARRATIVE_PATH" \
            --needs_file "$need_folder/info_need.json" \
            --output_file "$JUDGE_FOLDER/info_judge_end_time_output.json" \
            --model_id "$model_id" \
            --chunk_size 1 \
            --n_last 5 \
            --n_next 5 \
            --mode end_time &

        echo "[$need_folder| model: $model_id] Running user_information_need_simulation.llm_judge on video_need"
        python3 -m user_information_need_simulation.llm_judge \
            --transcript_file "$NARRATIVE_PATH" \
            --narrative_file "$NARRATIVE_PATH" \
            --needs_file "$need_folder/video_need.json" \
            --output_file "$JUDGE_FOLDER/video_judge_end_time_output.json" \
            --model_id "$model_id" \
            --chunk_size 1 \
            --n_last 5 \
            --n_next 5 \
            --mode end_time &
        # ((judge_index++))
        # done
        wait
    ) &
    pids+=($!)
done

# # # Wait for all top-level backgrounded subshells
for pid in "${pids[@]}"; do
    wait "$pid"
done  # Wait for all judging to finish

# # # Step 3: Parallelized end time finding
pids=()
for need_folder in "$PROJECT_FOLDER"/need_*; do
    for judge_folder in "$need_folder"/judge_time_*; do
        (
            for judge_file in "$judge_folder"/*.json; do
                base_file=$(basename "$judge_file")
                if [[ "$base_file" == *"info_judge_end_time_output.json" ]]; then
                    SENTENCES="$TRANSCRIPT_PATH"
                else
                    SENTENCES="$NARRATIVE_PATH"
                fi
                echo "[$judge_folder] Running user_information_need_simulation.end_time_finder on [$base_file]"
                output_file="${judge_file%_output.json}_updated_output.json"
                python3 -m user_information_need_simulation.end_time_finder \
                    --judge_output "$judge_file" \
                    --sentences "$SENTENCES" \
                    --output "$output_file"
            done
        ) &
        pids+=($!)
    done
done
wait  # Wait for all user_information_need_simulation.end_time_finder jobs

# # # # Wait for all backgrounded subshells to complete
for pid in "${pids[@]}"; do
    wait "$pid"
done

# # # Step 4: Aggregator step — single job
echo "[FINAL] Running aggregator"
for need_folder in "$PROJECT_FOLDER"/need_*; do
    python3 -m user_information_need_simulation.aggregator \
        --output "$need_folder/"aggregated.json \
        --input_folder "$need_folder/" 
done
wait

# # # # # Step 4: Deduplication step — single job
echo "[FINAL] Running deduplicator"
python3 -m user_information_need_simulation.deduplicator \
    --output "$PROJECT_FOLDER/"deduplicated.json \
    --input_folder "$PROJECT_FOLDER/" \
    --threshold 0.75
wait

# # # # Step 5: Judging end_time for missing ones
echo "Running user_information_need_simulation.llm_judge_time to find missing end times"
python3 -m user_information_need_simulation.llm_judge_time \
    --transcript_file "$TRANSCRIPT_PATH" \
    --narrative_file "$NARRATIVE_PATH" \
    --needs_file "$PROJECT_FOLDER/deduplicated.json" \
    --output_file "$PROJECT_FOLDER/deduplicated_updated.json" \
    --model_id "${JUDGE_END_TIME_MODEL_IDS[@]}" \
    --chunk_size 1 \
    --n_last 5 \
    --n_next 5
wait

# Step 5: Parallelize judging relevance
pids=()
judge_index=1
for model_id in "${JUDGE_SCORE_MODEL_IDS[@]}"; do
    JUDGE_FOLDER="$PROJECT_FOLDER/judge_score_$judge_index"
    mkdir -p "$JUDGE_FOLDER"
    echo "[$JUDGE_FOLDER | model: $model_id] Running user_information_need_simulation.llm_judge"

    python3 -m user_information_need_simulation.llm_judge \
        --transcript_file "$TRANSCRIPT_PATH" \
        --narrative_file "$NARRATIVE_PATH" \
        --needs_file "$PROJECT_FOLDER/deduplicated_updated.json" \
        --output_file "$JUDGE_FOLDER/judge_deduplicated.json" \
        --model_id "$model_id" \
        --chunk_size 1 \
        --n_last 5 \
        --n_next 5 \
        --mode score &
    pids+=($!)
    ((judge_index++))
done

# # # Wait for all backgrounded subshells (need_folders)
for pid in "${pids[@]}"; do
    wait "$pid"
done

# # # Step 6: Voting Aggregator
echo "[$PROJECT_FOLDER] Running voting_aggregator"
python3 -m user_information_need_simulation.voting_aggregator \
    --folder_path "$PROJECT_FOLDER/" \
    --score_threshold 6 \
    --voting_threshold 2

# # # Step 7: Aggregate Human Annotations
# echo "[$PROJECT_FOLDER] Running human_aggregator"
# python3 -m user_information_need_simulation.human_aggregator \
#     --output_file "$PROJECT_FOLDER/"final_aggr_output.json \
#     --input_file "$PROJECT_FOLDER/final_output.json" \
#     --input_human_file "$PROJECT_FOLDER/final_output.json" \
#     --threshold 0.75