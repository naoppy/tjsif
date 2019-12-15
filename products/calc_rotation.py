from motor import control_stepping_motor as motorCtl


def rotate(persons, width):
    # mean = statistics.mean([(person.bbox.xmax - person.bbox.min) / 2 for person in persons])
    # if mean > width / 2:
    #     # move camera right
    #     motorCtl.right_spin_7_2degree()
    # else:
    #     # move camera left
    #     motorCtl.left_spin_7_2degree()
    minX = min([person.bbox.xmin for person in persons])
    maxX = min([person.bbox.xmax for person in persons])
    left_diff = minX
    right_diff = width - maxX
    if left_diff > right_diff:
        # rotate right
        motorCtl.right_spin_7_2degree()
    else:
        motorCtl.left_spin_7_2degree()
