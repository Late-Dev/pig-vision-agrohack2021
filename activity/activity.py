def get_activity(track_points):
    if len(track_points) < 2:
        return 0
    last_points = track_points[-10:]
    mean_length = sum([
        ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5
        for p1, p2 in zip(last_points[:-1], last_points[1:])
    ]) // (len(last_points) - 1)
    return mean_length
