#!/bin/bash

# Automated rollout script for OS_Teleop_Extension
# Usage: ./rollout.sh [demo_name] [num_rollouts]

# Set defaults
DEMO_NAME="${1:-demo_01}"
NUM_ROLLOUTS="${2:-1}"
BASE_DIR="/home/stanford/OS_Teleop_Extension"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to wait for trigger file removal
wait_for_trigger_removal() {
    local trigger_file=$1
    local timeout=${2:-300}
    local elapsed=0
    
    echo "Waiting for process to complete..."
    while [ -f "$trigger_file" ] && [ $elapsed -lt $timeout ]; do
        sleep 1
        ((elapsed++))
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "  Still waiting... (${elapsed}s elapsed)"
        fi
    done
    
    if [ -f "$trigger_file" ]; then
        print_error "Timeout after ${timeout}s"
        return 1
    else
        print_success "Process completed in ${elapsed}s"
        return 0
    fi
}

# Function to generate cumulative success tally
generate_cumulative_tally() {
    echo ""
    echo "======================================================"
    echo "GENERATING CUMULATIVE SUCCESS TALLY FOR ALL DEMOS"
    echo "======================================================"
    
    # Initialize counters
    TOTAL_DEMOS=0
    TOTAL_SUCCESS=0
    RED_SUCCESS=0
    GREEN_SUCCESS=0
    BLUE_SUCCESS=0
    
    # Create tally file
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    TALLY_FILE="$BASE_DIR/rollout_tally_cumulative_${TIMESTAMP}.txt"
    
    # Header for tally file
    echo "Cumulative Rollout Success Tally - Generated at $(date)" > "$TALLY_FILE"
    echo "======================================" >> "$TALLY_FILE"
    echo "" >> "$TALLY_FILE"
    
    # Process ALL demo folders
    for demo_dir in "$BASE_DIR"/demo_*/; do
        if [ -d "$demo_dir" ]; then
            DEMO_NAME=$(basename "$demo_dir")
            SUCCESS_FILE="${demo_dir}success.json"
            
            if [ -f "$SUCCESS_FILE" ]; then
                TOTAL_DEMOS=$((TOTAL_DEMOS + 1))
                
                # Extract success values using grep and sed
                SUCCESS=$(grep -o '"success": [^,}]*' "$SUCCESS_FILE" | sed 's/"success": //')
                RED_CONTACT=$(grep -o '"red_white_contact": [^,}]*' "$SUCCESS_FILE" | sed 's/"red_white_contact": //')
                GREEN_CONTACT=$(grep -o '"green_white_contact": [^,}]*' "$SUCCESS_FILE" | sed 's/"green_white_contact": //')
                BLUE_CONTACT=$(grep -o '"blue_white_contact": [^,}]*' "$SUCCESS_FILE" | sed 's/"blue_white_contact": //')
                
                # Update counters
                if [ "$SUCCESS" = "true" ]; then
                    TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
                fi
                if [ "$RED_CONTACT" = "true" ]; then
                    RED_SUCCESS=$((RED_SUCCESS + 1))
                fi
                if [ "$GREEN_CONTACT" = "true" ]; then
                    GREEN_SUCCESS=$((GREEN_SUCCESS + 1))
                fi
                if [ "$BLUE_CONTACT" = "true" ]; then
                    BLUE_SUCCESS=$((BLUE_SUCCESS + 1))
                fi
                
                # Log to file
                echo "$DEMO_NAME: Success=$SUCCESS, Red=$RED_CONTACT, Green=$GREEN_CONTACT, Blue=$BLUE_CONTACT" >> "$TALLY_FILE"
            fi
        fi
    done
    
    # Calculate percentages
    if [ $TOTAL_DEMOS -gt 0 ]; then
        SUCCESS_PCT=$(awk "BEGIN {printf \"%.1f\", $TOTAL_SUCCESS / $TOTAL_DEMOS * 100}")
        RED_PCT=$(awk "BEGIN {printf \"%.1f\", $RED_SUCCESS / $TOTAL_DEMOS * 100}")
        GREEN_PCT=$(awk "BEGIN {printf \"%.1f\", $GREEN_SUCCESS / $TOTAL_DEMOS * 100}")
        BLUE_PCT=$(awk "BEGIN {printf \"%.1f\", $BLUE_SUCCESS / $TOTAL_DEMOS * 100}")
    else
        SUCCESS_PCT="0.0"
        RED_PCT="0.0"
        GREEN_PCT="0.0"
        BLUE_PCT="0.0"
    fi
    
    # Summary section
    echo "" >> "$TALLY_FILE"
    echo "SUMMARY" >> "$TALLY_FILE"
    echo "=======" >> "$TALLY_FILE"
    echo "Total Demos: $TOTAL_DEMOS" >> "$TALLY_FILE"
    echo "Total Success (all 3 blocks): $TOTAL_SUCCESS/$TOTAL_DEMOS ($SUCCESS_PCT%)" >> "$TALLY_FILE"
    echo "Red Block Success: $RED_SUCCESS/$TOTAL_DEMOS ($RED_PCT%)" >> "$TALLY_FILE"
    echo "Green Block Success: $GREEN_SUCCESS/$TOTAL_DEMOS ($GREEN_PCT%)" >> "$TALLY_FILE"
    echo "Blue Block Success: $BLUE_SUCCESS/$TOTAL_DEMOS ($BLUE_PCT%)" >> "$TALLY_FILE"
    
    # Print summary to console
    echo ""
    echo "CUMULATIVE SUCCESS TALLY SUMMARY"
    echo "================================"
    echo "Total Demos Found: $TOTAL_DEMOS"
    echo "Total Success (all 3 blocks): $TOTAL_SUCCESS/$TOTAL_DEMOS ($SUCCESS_PCT%)"
    echo "Red Block Success: $RED_SUCCESS/$TOTAL_DEMOS ($RED_PCT%)"
    echo "Green Block Success: $GREEN_SUCCESS/$TOTAL_DEMOS ($GREEN_PCT%)"
    echo "Blue Block Success: $BLUE_SUCCESS/$TOTAL_DEMOS ($BLUE_PCT%)"
    echo ""
    print_success "Detailed results saved to: $TALLY_FILE"
}

# Function to generate success tally
generate_success_tally() {
    echo ""
    echo "======================================================"
    echo "GENERATING SUCCESS TALLY"
    echo "======================================================"
    
    # Initialize counters
    TOTAL_DEMOS=0
    TOTAL_SUCCESS=0
    RED_SUCCESS=0
    GREEN_SUCCESS=0
    BLUE_SUCCESS=0
    
    # Create tally file
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    TALLY_FILE="$BASE_DIR/rollout_tally_${TIMESTAMP}.txt"
    
    # Header for tally file
    echo "Rollout Success Tally - Generated at $(date)" > "$TALLY_FILE"
    echo "======================================" >> "$TALLY_FILE"
    echo "" >> "$TALLY_FILE"
    
    # Process each demo folder
    for demo_dir in "$BASE_DIR"/demo_*/; do
        if [ -d "$demo_dir" ]; then
            DEMO_NAME=$(basename "$demo_dir")
            SUCCESS_FILE="${demo_dir}success.json"
            
            if [ -f "$SUCCESS_FILE" ]; then
                TOTAL_DEMOS=$((TOTAL_DEMOS + 1))
                
                # Extract success values using grep and sed
                SUCCESS=$(grep -o '"success": [^,}]*' "$SUCCESS_FILE" | sed 's/"success": //')
                RED_CONTACT=$(grep -o '"red_white_contact": [^,}]*' "$SUCCESS_FILE" | sed 's/"red_white_contact": //')
                GREEN_CONTACT=$(grep -o '"green_white_contact": [^,}]*' "$SUCCESS_FILE" | sed 's/"green_white_contact": //')
                BLUE_CONTACT=$(grep -o '"blue_white_contact": [^,}]*' "$SUCCESS_FILE" | sed 's/"blue_white_contact": //')
                
                # Update counters
                if [ "$SUCCESS" = "true" ]; then
                    TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
                fi
                if [ "$RED_CONTACT" = "true" ]; then
                    RED_SUCCESS=$((RED_SUCCESS + 1))
                fi
                if [ "$GREEN_CONTACT" = "true" ]; then
                    GREEN_SUCCESS=$((GREEN_SUCCESS + 1))
                fi
                if [ "$BLUE_CONTACT" = "true" ]; then
                    BLUE_SUCCESS=$((BLUE_SUCCESS + 1))
                fi
                
                # Log to file
                echo "$DEMO_NAME: Success=$SUCCESS, Red=$RED_CONTACT, Green=$GREEN_CONTACT, Blue=$BLUE_CONTACT" >> "$TALLY_FILE"
            fi
        fi
    done
    
    # Calculate percentages
    if [ $TOTAL_DEMOS -gt 0 ]; then
        SUCCESS_PCT=$(awk "BEGIN {printf \"%.1f\", $TOTAL_SUCCESS / $TOTAL_DEMOS * 100}")
        RED_PCT=$(awk "BEGIN {printf \"%.1f\", $RED_SUCCESS / $TOTAL_DEMOS * 100}")
        GREEN_PCT=$(awk "BEGIN {printf \"%.1f\", $GREEN_SUCCESS / $TOTAL_DEMOS * 100}")
        BLUE_PCT=$(awk "BEGIN {printf \"%.1f\", $BLUE_SUCCESS / $TOTAL_DEMOS * 100}")
    else
        SUCCESS_PCT="0.0"
        RED_PCT="0.0"
        GREEN_PCT="0.0"
        BLUE_PCT="0.0"
    fi
    
    # Summary section
    echo "" >> "$TALLY_FILE"
    echo "SUMMARY" >> "$TALLY_FILE"
    echo "=======" >> "$TALLY_FILE"
    echo "Total Demos: $TOTAL_DEMOS" >> "$TALLY_FILE"
    echo "Total Success (all 3 blocks): $TOTAL_SUCCESS/$TOTAL_DEMOS ($SUCCESS_PCT%)" >> "$TALLY_FILE"
    echo "Red Block Success: $RED_SUCCESS/$TOTAL_DEMOS ($RED_PCT%)" >> "$TALLY_FILE"
    echo "Green Block Success: $GREEN_SUCCESS/$TOTAL_DEMOS ($GREEN_PCT%)" >> "$TALLY_FILE"
    echo "Blue Block Success: $BLUE_SUCCESS/$TOTAL_DEMOS ($BLUE_PCT%)" >> "$TALLY_FILE"
    
    # Print summary to console
    echo ""
    echo "SUCCESS TALLY SUMMARY"
    echo "===================="
    echo "Total Demos: $TOTAL_DEMOS"
    echo "Total Success (all 3 blocks): $TOTAL_SUCCESS/$TOTAL_DEMOS ($SUCCESS_PCT%)"
    echo "Red Block Success: $RED_SUCCESS/$TOTAL_DEMOS ($RED_PCT%)"
    echo "Green Block Success: $GREEN_SUCCESS/$TOTAL_DEMOS ($GREEN_PCT%)"
    echo "Blue Block Success: $BLUE_SUCCESS/$TOTAL_DEMOS ($BLUE_PCT%)"
    echo ""
    print_success "Detailed results saved to: $TALLY_FILE"
}

# Function to run a single rollout
run_rollout() {
    local demo_name=$1
    
    echo ""
    echo "======================================================"
    echo "Running Rollout: $demo_name"
    echo "======================================================"
    
    # Step 1: Reset
    print_step "Step 1: Resetting Environment"
    cd "$BASE_DIR"
    touch reset_trigger.txt
    print_success "Created reset_trigger.txt"
    
    if wait_for_trigger_removal "reset_trigger.txt" 60; then
        print_success "Reset completed"
    fi
    sleep 2
    
    # Step 2: Data Collection - Trigger without waiting
    print_step "Step 2: Triggering Data Collection"
    cd "$BASE_DIR"
    touch "log_trigger_${demo_name}.txt"
    print_success "Created log_trigger_${demo_name}.txt"
    
    # Wait briefly for data collection to start
    sleep 2
    
    # Step 3: Model Trigger - Immediately after data collection
    print_step "Step 3: Triggering Model (immediately)"
    cd "$BASE_DIR"
    touch model_trigger.txt
    print_success "Created model_trigger.txt"
    
    echo "Data collection running (fixed duration ~22s)..."
    echo "Model triggered and waiting for data..."
    
    # Wait for model trigger to be consumed (indicates model has started)
    ELAPSED=0
    while [ -f "model_trigger.txt" ] && [ $ELAPSED -lt 30 ]; do
        sleep 0.5
        ((ELAPSED++))
    done
    
    if [ ! -f "model_trigger.txt" ]; then
        print_success "Model has started execution"
    fi
    
    # Wait for success.json to be created (indicates completion)
    echo "Waiting for model to complete and save results..."
    SUCCESS_FILE="${BASE_DIR}/${demo_name}/success.json"
    WAIT_TIME=0
    
    while [ ! -f "$SUCCESS_FILE" ] && [ $WAIT_TIME -lt 300 ]; do
        sleep 1
        ((WAIT_TIME++))
        if [ $((WAIT_TIME % 10)) -eq 0 ]; then
            echo "  Still waiting... (${WAIT_TIME}s elapsed)"
        fi
    done
    
    if [ -f "$SUCCESS_FILE" ]; then
        print_success "Model execution completed - success.json found"
    else
        print_error "Warning: Model execution may not have completed (no success.json found)"
    fi
    
    sleep 2
    
    echo ""
    print_success "Rollout $demo_name completed!"
}

# Main execution
echo "======================================================"
echo "OS_Teleop_Extension Automated Rollout Script"
echo "======================================================"
echo "Demo: $DEMO_NAME"
echo "Number of rollouts: $NUM_ROLLOUTS"
echo "Base directory: $BASE_DIR"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate orbitsurgical

# Run rollouts
if [ "$NUM_ROLLOUTS" -eq 1 ]; then
    run_rollout "$DEMO_NAME"
else
    # Extract demo prefix and starting number
    DEMO_PREFIX=$(echo "$DEMO_NAME" | sed 's/_[0-9]*$//')
    START_NUM=$(echo "$DEMO_NAME" | grep -o '[0-9]*$')
    START_NUM=${START_NUM:-1}
    
    for ((i=0; i<NUM_ROLLOUTS; i++)); do
        CURRENT_NUM=$((START_NUM + i))
        CURRENT_DEMO="${DEMO_PREFIX}_$(printf "%02d" $CURRENT_NUM)"
        
        echo ""
        echo "======================================================"
        echo "ROLLOUT $((i+1))/$NUM_ROLLOUTS"
        echo "======================================================"
        
        run_rollout "$CURRENT_DEMO"
        
        if [ $((i+1)) -lt $NUM_ROLLOUTS ]; then
            echo ""
            echo "Waiting 5s before next rollout..."
            sleep 5
        fi
    done
fi

echo ""
echo "======================================================"
print_success "All rollouts completed!"
echo "======================================================"

# Generate success tally
generate_success_tally 