"""FIFA standard pitch dimensions and named landmark coordinates."""

# Standard FIFA pitch dimensions (meters)
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

# Named landmarks: (x, y) in meters.
# Origin (0, 0) = bottom-left corner (left goal-line / near touchline).
# X-axis = along length (0..105), Y-axis = along width (0..68).
LANDMARKS = {
    "bottom_left_corner": (0.0, 0.0),
    "top_left_corner": (0.0, 68.0),
    "bottom_right_corner": (105.0, 0.0),
    "top_right_corner": (105.0, 68.0),
    "halfway_line_bottom": (52.5, 0.0),
    "halfway_line_top": (52.5, 68.0),
    "center_circle_center": (52.5, 34.0),
    "left_goal_center": (0.0, 34.0),
    "right_goal_center": (105.0, 34.0),
    "left_penalty_spot": (11.0, 34.0),
    "right_penalty_spot": (94.0, 34.0),
    "left_penalty_area_top_left": (0.0, 54.16),
    "left_penalty_area_top_right": (16.5, 54.16),
    "left_penalty_area_bottom_left": (0.0, 13.84),
    "left_penalty_area_bottom_right": (16.5, 13.84),
    "right_penalty_area_top_left": (88.5, 54.16),
    "right_penalty_area_top_right": (105.0, 54.16),
    "right_penalty_area_bottom_left": (88.5, 13.84),
    "right_penalty_area_bottom_right": (105.0, 13.84),
    "left_goal_area_top_right": (5.5, 43.16),
    "left_goal_area_bottom_right": (5.5, 24.84),
    "right_goal_area_top_left": (99.5, 43.16),
    "right_goal_area_bottom_left": (99.5, 24.84),
}
