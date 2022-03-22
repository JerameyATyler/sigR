class RNG:

    def __init__(self, seed='0xec0ec0', duration=10, delay_limits=(1, 60), time_limits=(1, 8), reflection_limits=(4, 8), zenith_limits=(-40, 90), azimuth_limits=(0, 360)):
        import numpy as np
        from data_loader import list_hrtf_data, list_anechoic_data, list_composers, get_hrtfs

        self.seed = seed
        self.duration = duration
        self.delay_limits = delay_limits
        self.time_limits = time_limits
        self.reflection_limits = reflection_limits
        self.zenith_limits = zenith_limits
        self.azimuth_limits = azimuth_limits

        self.rng = np.random.default_rng(int(self.seed, 0))

        self.composers = list_composers()
        self.anechoic_data = list_anechoic_data()

        if zenith_limits is not None:
            zmin, zmax = zenith_limits
        else:
            zmin, zmax = None, None
        if azimuth_limits is not None:
            amin, amax = azimuth_limits
        else:
            amin, amax = None, None

        hrtf_data = list_hrtf_data()
        zeniths, azimuths = get_hrtfs(amin=amin, amax=amax, zmin=zmin, zmax=zmax)
        hrtfs = {z: {} for z in zeniths}
        for z in zeniths:
            for a in azimuths:
                if a in hrtf_data[z].keys():
                    hrtfs[z][a] = hrtf_data[z][a]

        self.hrtf_data = hrtfs

    def get_composer(self):
        return self.rng.choice(self.composers)

    def get_part_count(self, composer):
        parts = self.anechoic_data[composer]
        part_limits = (2, len(parts))
        return self.rng.integers(part_limits[0], part_limits[1])

    def get_parts(self, composer=None, part_count=None):
        if composer is None:
            composer = self.get_composer()
        if part_count is None:
            part_count = self.get_part_count(composer)

        return self.rng.choice(self.anechoic_data[composer], part_count, replace=False)

    def get_zenith(self, azimuth=None):
        zeniths = sorted(list(self.hrtf_data.keys()))
        if azimuth is not None:
            zeniths = [z for z in zeniths if azimuth in self.hrtf_data[z]]

        return self.rng.choice(zeniths)

    def get_azimuth(self, zenith=None):
        zeniths = []
        if zenith is not None:
            zeniths .append(zenith)
        else:
            zeniths = sorted(list(self.hrtf_data.keys()))

        azimuths = set()
        for z in zeniths:
            for a in self.hrtf_data[z]:
                azimuths.add(a)

        return self.rng.choice(list(azimuths))

    def get_delay(self):
        return self.rng.integers(low=self.delay_limits[0], high=self.delay_limits[1] + 1)

    def get_amplitude(self):
        return self.rng.random()

    def get_time(self):
        return self.rng.integers(low=self.time_limits[0], high=self.time_limits[1] + 1)

    def get_reflection_count(self):
        return self.rng.integers(low=self.reflection_limits[0], high=self.reflection_limits[1] + 1)

    def get_offset(self, length):
        length = length - self.duration * 1000
        return self.rng.integers(low=0, high=length)
