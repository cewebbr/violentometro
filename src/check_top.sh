
# Load manually saved process IDs:
capture_pid=`cat last_capture_pid`
rating_pid=`cat last_rating_pid`


# Capture status:
echo "CAPTURE SYSTEM"
echo "    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND"
top -b -n 1 -p $capture_pid 2>/dev/null | grep hxavier

echo ""

# AI rating status:
echo "RATING SYSTEM"
echo "    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND"
top -b -n 1 -p $rating_pid 2>/dev/null | grep hxavier
