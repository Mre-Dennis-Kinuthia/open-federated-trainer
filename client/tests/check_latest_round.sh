#!/bin/bash
# Check the latest active round

# Try to find the latest round by checking rounds in reverse
for round_id in {10..1}; do
    status=$(curl -s http://localhost:8000/status/$round_id 2>/dev/null)
    if [ $? -eq 0 ] && [ -n "$status" ] && ! echo "$status" | grep -q "not found"; then
        echo "Latest Round: $round_id"
        echo "$status" | python3 -m json.tool
        exit 0
    fi
done

echo "No active rounds found"

