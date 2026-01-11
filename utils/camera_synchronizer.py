class CameraSynchronizer:
    def __init__(self, master_camera, slave_cameras, max_time_diff_ms=40):
        self.master = master_camera
        self.slaves = (
            slave_cameras if isinstance(slave_cameras, list) else [slave_cameras]
        )
        self.max_time_diff = max_time_diff_ms / 1000.0  # Convert to seconds

    def get_synchronized_frames(self):
        master_frame, master_ts = self.master.get_frame()

        if master_frame is None or master_ts is None:
            return None

        slave_frames = []
        time_diffs = []
        synced = True

        for slave in self.slaves:
            slave_frame, slave_ts, time_diff = slave.get_nearest_frame(master_ts)

            if slave_frame is None:
                return None

            slave_frames.append((slave_frame, slave_ts))
            time_diffs.append(time_diff * 1000)

            if time_diff > self.max_time_diff:
                synced = False

        return {
            "master": (master_frame, master_ts),
            "slaves": slave_frames,
            "synced": synced,
            "time_diffs": time_diffs,
        }

    def get_synced_pair(self):
        result = self.get_synchronized_frames()

        if result is None:
            return None, None, False, 0

        master_frame, master_ts = result["master"]
        slave_frame, slave_ts = result["slaves"][0]
        synced = result["synced"]
        time_diff_ms = result["time_diffs"][0]

        return master_frame, slave_frame, synced, time_diff_ms
