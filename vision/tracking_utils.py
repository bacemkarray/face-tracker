import time
from typing import Tuple

import cv2

from ultralytics.utils.plotting import Annotator, colors
import interactive_tracker as it


def get_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """
    Calculate the center point of a bounding box.

    Args:
        x1 (int): Top-left X coordinate.
        y1 (int): Top-left Y coordinate.
        x2 (int): Bottom-right X coordinate.
        y2 (int): Bottom-right Y coordinate.

    Returns:
        center_x (int): X-coordinate of the center point.
        center_y (int): Y-coordinate of the center point.
    """
    return (x1 + x2) // 2, (y1 + y2) // 2


def extend_line_from_edge(mid_x: int, mid_y: int, direction: str, img_shape: Tuple[int, int, int]) -> Tuple[int, int]:
    """
    Calculate the endpoint to extend a line from the center toward an image edge.

    Args:
        mid_x (int): X-coordinate of the midpoint.
        mid_y (int): Y-coordinate of the midpoint.
        direction (str): Direction to extend ('left', 'right', 'up', 'down').
        img_shape (Tuple[int, int, int]): Image shape in (height, width, channels).

    Returns:
        end_x (int): X-coordinate of the endpoint.
        end_y (int): Y-coordinate of the endpoint.
    """
    h, w = img_shape[:2]
    if direction == "left":
        return 0, mid_y
    if direction == "right":
        return w - 1, mid_y
    if direction == "up":
        return mid_x, 0
    if direction == "down":
        return mid_x, h - 1
    return mid_x, mid_y


def draw_tracking_scope(im, bbox: tuple, color: tuple) -> None:
    """
    Draw tracking scope lines extending from the bounding box to image edges.

    Args:
        im (ndarray): Image array to draw on.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        color (tuple): Color in BGR format for drawing.
    """
    x1, y1, x2, y2 = bbox
    mid_top = ((x1 + x2) // 2, y1)
    mid_bottom = ((x1 + x2) // 2, y2)
    mid_left = (x1, (y1 + y2) // 2)
    mid_right = (x2, (y1 + y2) // 2)
    cv2.line(im, mid_top, extend_line_from_edge(*mid_top, "up", im.shape), color, 2)
    cv2.line(im, mid_bottom, extend_line_from_edge(*mid_bottom, "down", im.shape), color, 2)
    cv2.line(im, mid_left, extend_line_from_edge(*mid_left, "left", im.shape), color, 2)
    cv2.line(im, mid_right, extend_line_from_edge(*mid_right, "right", im.shape), color, 2)


def draw_dashed_box(im, x1, y1, x2, y2, label, color, txt_color):
    # Draw dashed box for other objects
            for i in range(x1, x2, 10):
                cv2.line(im, (i, y1), (i + 5, y1), color, 3)
                cv2.line(im, (i, y2), (i + 5, y2), color, 3)
            for i in range(y1, y2, 10):
                cv2.line(im, (x1, i), (x1, i + 5), color, 3)
                cv2.line(im, (x2, i), (x2, i + 5), color, 3)
            # Draw label text with background
            (tw, th), bl = cv2.getTextSize(label, 0, 0.7, 2)
            cv2.rectangle(im, (x1 + 5 - 5, y1 + 20 - th - 5), (x1 + 5 + tw + 5, y1 + 20 + bl), color, -1)
            cv2.putText(im, label, (x1 + 5, y1 + 20), 0, 0.7, txt_color, 1, cv2.LINE_AA)


def process_detections(frame, detections, selected_id, show_conf, class_names):
    annotator = Annotator(frame)    
    center = None

    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue

        x1, y1, x2, y2 = map(int, track[:4])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1
        color = colors(track_id, True)
        txt_color = annotator.get_txt_color(color)
        label = f"{class_names[class_id]} ID {track_id}" + (f" ({float(track[5]):.2f})" if show_conf else "")
        # if the face being tracked is equal to the face the user selected.
        if track_id == selected_id:
            it.draw_tracking_scope(frame, (x1, y1, x2, y2), color)
            center = it.get_center(x1, y1, x2, y2)
            cv2.circle(frame, center, 6, color, -1)

            # Pulsing circle for attention
            pulse_radius = 8 + int(4 * abs(time.time() % 1 - 0.5))
            cv2.circle(frame, center, pulse_radius, color, 2)

            annotator.box_label([x1, y1, x2, y2], label=f"ACTIVE: TRACK {track_id}", color=color)
        else:
            draw_dashed_box(frame, x1, y1, x2, y2, label, color, txt_color)
    return frame, center