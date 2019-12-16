from motor import control_stepping_motor as motorCtl
import time

last_move_time = time.time()
last_move_direction = "L"


def rotate(persons, width):
    if not persons:
        return

    global last_move_time
    now = time.time()

    if now - last_move_time < 2:
        return

    min_x = min([person.bbox.xmin for person in persons])
    max_x = min([person.bbox.xmax for person in persons])
    left_diff = min_x
    right_diff = width - max_x

    if left_diff == right_diff:
        return

    move_direction = "R" if left_diff > right_diff else "L"

    # Reduce left and right round trip
    global last_move_direction
    if now - last_move_time < 3 and last_move_direction != move_direction:
        return

    if move_direction == "R":
        # rotate right
        last_move_time = now
        last_move_direction = "R"
        motorCtl.right_spin_7_2degree()
    else:
        # rotate left
        last_move_time = now
        last_move_direction = "L"
        motorCtl.left_spin_7_2degree()
