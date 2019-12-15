from motor import control_stepping_motor as motorCtl
import time

last_move_time = time.time()
last_move_direction = "L"


def rotate(persons, width):
    if not persons:
        return

    global last_move_time
    now = time.time()

    min_x = min([person.bbox.xmin for person in persons])
    max_x = min([person.bbox.xmax for person in persons])
    left_diff = min_x
    right_diff = width - max_x

    move_direction = "R" if left_diff > right_diff else "L"

    # Reduce left and right round trip
    if now - last_move_time < 2 and last_move_direction != move_direction:
        return

    if left_diff > right_diff:
        # rotate right
        last_move_time = now
        motorCtl.right_spin_7_2degree()
    else:
        last_move_time = now
        motorCtl.left_spin_7_2degree()
