#!/bin/bash

# This script configures the display resolution for a Luckfox Debian system.
# It sets a custom resolution of 1920x550, enables HDMI-1, and disables DSI-1 and DP-1.

# Define the desired resolution and refresh rate
WIDTH=1920
HEIGHT=550

echo "Attempting to configure display for resolution ${WIDTH}x${HEIGHT}."

# Step 1: Define the modeline for the custom resolution.
# This modeline was generated using the 'cvt' command for 1920x550 @ 60Hz.
MODELINE_VALUES="1920x550_60.00 59.81 1920 1968 2000 2064 550 553 558 560 -hsync +vsync"

# The `xrandr --newmode` command adds a new video mode.
echo "Adding new modeline: ${MODELINE_VALUES}"
xrandr --newmode ${MODELINE_VALUES}

# Step 2: Add the new mode to the HDMI-1 output.
# The `xrandr --addmode` command assigns the new mode to a specific display output.
echo "Adding mode to HDMI-1..."
xrandr --addmode HDMI-1 "1920x550_60.00"

# Step 3: Set HDMI-1 to the new resolution and disable other outputs.
# The `xrandr --output` command is used to configure individual outputs.
# `--mode` sets the resolution.
# `--off` disables the output.
echo "Setting HDMI-1 to ${WIDTH}x${HEIGHT} and disabling DSI-1 and DP-1..."
xrandr --output HDMI-1 --mode "1920x550_60.00" --output DSI-1 --off --output DP-1 --off

echo "Display configuration complete. Please check your screen."
