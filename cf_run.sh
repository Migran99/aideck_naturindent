session=${USER}_AI


# Create new session  (-2 allows 256 colors in the terminal, -s -> session name, -d -> not attach to the new session)
tmux -2 new-session -d -s $session

# Create roscore 
# send-keys writes the string into the sesssion (-t -> target session , C-m -> press Enter Button)
tmux new-window -t $session:0 -n 'stream_pub'
tmux send-keys -t $session:0 "ros2 launch aideck_stream_publisher aideck_stream.py ip:=192.168.2.21 show_flag:=False" C-m

tmux new-window -t $session:1 -n 'virtualcam'
tmux send-keys -t $session:1 "ros2 launch ros2_virtualcam virtualcam_launch.py" C-m

echo Finished!

tmux attach -t $session:0