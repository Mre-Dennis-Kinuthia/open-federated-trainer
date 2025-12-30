#!/bin/bash
# Monitor all active federated learning rounds

echo "Monitoring Federated Learning Rounds..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Federated Learning Round Status"
    echo "Time: $(date '+%H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check rounds 1-10
    for round_id in {1..10}; do
        status=$(curl -s http://localhost:8000/status/$round_id 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$status" ] && ! echo "$status" | grep -q "not found"; then
            echo "Round $round_id:"
            echo "$status" | python3 -m json.tool 2>/dev/null | grep -E "(round_id|state|total_clients|total_updates)" | sed 's/^/  /'
            echo ""
        fi
    done
    
    sleep 2
done

