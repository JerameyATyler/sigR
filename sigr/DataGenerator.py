class DataGenerator:
    def __init__(self, sample_count, output_directory, rng=None, fs=24000, verbose=False):
        import os
        import hashlib
        from pathlib import Path

        if rng is None:
            from RNG import RNG
            rng = RNG()
        self.rng = rng
        self.sample_count = sample_count
        self.recipe = None
        self.chunk_size = 50
        self.fs = fs
        self.verbose = verbose

        path = Path(output_directory)
        if not os.path.isdir(path):
            os.mkdir(path)
        path = path / 'reflections'
        if not os.path.isdir(path):
            os.mkdir(path)
        self.output_directory = path.__str__()

        s = f'{rng.seed}{rng.duration}{rng.delay_limits}{rng.time_limits}{rng.reflection_limits}{rng.zenith_limits}' \
            f'{rng.azimuth_limits}{sample_count}{verbose}'
        self.hash = str(int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

    def generate(self):
        import pandas as pd
        import dask.dataframe as dd
        import dask
        from pathlib import Path
        import numpy as np
        import os
        from data_loader import read_recipe

        print('Data generator started')

        dfi = self.generate_ingredients_list()

        filepath = Path(self.output_directory) / f'recipe_{self.hash}'

        if not os.path.isdir(filepath):
            sample_count = self.sample_count
            chunk_size = self.chunk_size

            batches = int(np.ceil(sample_count / chunk_size))
            results = []
            print('Generating recipe batches')
            for i in range(batches):
                if (i + 1) * chunk_size > sample_count:
                    chunk = sample_count % chunk_size
                else:
                    chunk = chunk_size
                result = dask.delayed(self.generate_recipe)(chunk)
                results.append(result)

            df = pd.concat(dask.compute(*results))
            ddf = dd.from_pandas(df, chunksize=chunk_size)
            print('Writing recipes')
            ddf.to_parquet(filepath, engine='pyarrow')
            print('Generating samples')
            s = ddf.map_partitions(self.generate_samples, meta=ddf)
            s.compute()
            df = ddf.compute()
        else:
            df = read_recipe((Path(self.output_directory) / f'recipe_{self.hash}').__str__())
        return dfi, df

    def generate_ingredients_list(self):
        import pandas as pd
        from pathlib import Path
        import os

        print('Generating ingredients list')

        filepath = Path(self.output_directory) / f'ingredients_{self.hash}.json'

        rng = self.rng

        df = pd.DataFrame(dict(
            seed=rng.seed,
            duration=rng.duration,
            delay_limits=[rng.delay_limits],
            time_limits=[rng.time_limits],
            reflections_limits=[rng.reflection_limits],
            zenith_limits=[rng.zenith_limits],
            azimuth_limits=[rng.azimuth_limits],
            sample_count=self.sample_count
        ))

        if not os.path.isfile(filepath):
            df.to_json(filepath, orient='records', lines=True)
        return df

    def generate_recipe(self, count):
        import pandas as pd
        import dask

        print('Generating recipes')

        lazy_results = []
        for i in range(count):
            lazy_result = dask.delayed(self.generate_sample_recipe)()
            lazy_results.append(lazy_result)

        df = pd.DataFrame(dask.compute(*lazy_results))

        return df

    def generate_sample_recipe(self):
        from data_loader import list_anechoic_lengths
        import hashlib

        lengths = list_anechoic_lengths()
        rng = self.rng

        composer = rng.get_composer()
        part_count = rng.get_part_count(composer=composer)
        parts = rng.get_parts(composer=composer, part_count=part_count)
        offset = rng.get_offset(lengths[composer])
        duration = rng.duration
        zenith = rng.get_zenith()
        azimuth = rng.get_azimuth(zenith=zenith)
        reverb_time = rng.get_time()
        reverb_delay = rng.get_delay()
        reverb_amplitude = rng.rng.uniform(0, 0.05)
        reflection_count = rng.get_reflection_count()
        reflection_zenith, reflection_azimuth, reflection_amplitude, reflection_delay = self.get_reflections(
            reflection_count)
        s = f'{part_count:02d}{"".join(parts)}{offset}{zenith}{azimuth}{reflection_count}' \
            f'{"".join(str(x) for x in reflection_zenith)}{"".join(str(x) for x in reflection_azimuth)}' \
            f'{"".join(str(x) for x in reflection_amplitude)}{"".join(str(x) for x in reflection_delay)}' \
            f'{reverb_amplitude}{reverb_delay}{reverb_time}'

        s = str(int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8)
        filepath = f'{composer}_{s}'
        print(f'Generating recipe {filepath}\n')

        return dict(
            composer=composer,
            part_count=part_count,
            parts=parts,
            zenith=zenith,
            azimuth=azimuth,
            offset=offset,
            duration=duration,
            reverb_time=reverb_time,
            reverb_delay=reverb_delay,
            reverb_amplitude=reverb_amplitude,
            reflection_count=reflection_count,
            reflection_amplitude=reflection_amplitude,
            reflection_delay=reflection_delay,
            reflection_zenith=reflection_zenith,
            reflection_azimuth=reflection_azimuth,
            filepath=filepath,
            name=''
        )

    def generate_sample(self, recipe):
        from audio_processing import mix_parts, apply_hrtf, mix_reflections, apply_reverberation, sum_signals, \
            adjust_signal_to_noise
        from pathlib import Path
        from utils import generate_impulse
        import os
        from pydub import AudioSegment

        impulse = generate_impulse(recipe['duration'])

        print(f'Generating sample: {recipe["filepath"]}\n')

        print(f'\tMixing parts: {recipe["filepath"]}')
        filepath = Path(f"raw/{recipe['filepath']}_raw.wav")
        if os.path.isfile(filepath):
            signal = AudioSegment.from_wav(filepath)
        else:
            signal = mix_parts(recipe['parts'], recipe['offset'], recipe['duration'])

        print(f'\tApplying HRTF: {recipe["filepath"]}')
        filepath = Path(f'hrtf/{recipe["filepath"]}_hrtf.wav')
        if os.path.isfile(filepath):
            hrtf = AudioSegment.from_wav(filepath)
        else:
            hrtf = apply_hrtf(signal, recipe['zenith'], recipe['azimuth'])
        impulse_hrtf = apply_hrtf(impulse, recipe['zenith'], recipe['azimuth'])

        print(f'\tApplying reflections: {recipe["filepath"]}')
        filepath = Path(f"reflections/{recipe['filepath']}_reflections.wav")
        if os.path.isfile(filepath):
            reflections = AudioSegment.from_wav(filepath)
        else:
            reflections = mix_reflections(hrtf, recipe['reflection_count'], recipe['reflection_amplitude'],
                                          recipe['reflection_delay'], recipe['reflection_zenith'],
                                          recipe['reflection_azimuth'])
        impulse_reflections = mix_reflections(impulse_hrtf, recipe['reflection_count'],
                                              recipe['reflection_amplitude'], recipe['reflection_delay'],
                                              recipe['reflection_zenith'], recipe['reflection_azimuth'])

        print(f'\tApplying reverberation: {recipe["filepath"]}')
        filepath = Path(f"reverberation/{recipe['filepath']}_reverberation.wav")
        if os.path.isfile(filepath):
            reverberation = AudioSegment.from_wav(filepath)
        else:
            reverberation = apply_reverberation(hrtf, recipe['reverb_amplitude'], recipe['reverb_delay'],
                                                recipe['reverb_time'])
        impulse_reverberation = apply_reverberation(impulse_hrtf, recipe['reverb_amplitude'], recipe['reverb_delay'],
                                                    recipe['reverb_time'])

        print(f'\tSumming signals: {recipe["filepath"]}')
        filepath = Path(f"summation/{recipe['filepath']}_summation.wav")
        if os.path.isfile(filepath):
            summation = AudioSegment.from_wav(filepath)
        else:
            summation = sum_signals(reflections, reverberation)
        impulse_summation = sum_signals(impulse_reflections, impulse_reverberation)

        print(f'\tAdjusting signal-to-noise ratio: {recipe["filepath"]}')
        filepath = Path(f"noise/{recipe['filepath']}.wav")
        if os.path.isfile(filepath):
            noise = AudioSegment.from_wav(filepath)
        else:
            noise = adjust_signal_to_noise(summation, -60)
        impulse_noise = adjust_signal_to_noise(impulse_summation, -60)

        print(f'\tTrimming sample: {recipe["filepath"]}')
        filepath = Path(f"samples/{recipe['filepath']}.wav")
        if os.path.isfile(filepath):
            sample = AudioSegment.from_wav(filepath)
        else:
            sample = noise[:recipe['duration'] * 1000]
        impulse_sample = impulse_noise[:recipe["duration"] * 1000]

        self.write(sample, 'samples', f'{recipe["filepath"]}.wav')
        self.write(impulse_sample, 'rir', f'{recipe["filepath"]}_rir.wav')

        if self.verbose:
            self.write(signal, 'raw', f'{recipe["filepath"]}_raw.wav')
            self.write(hrtf, 'hrtf', f'{recipe["filepath"]}_hrtf.wav')
            self.write(reflections, 'reflections', f'{recipe["filepath"]}_reflections.wav')
            self.write(reverberation, 'reverberation', f'{recipe["filepath"]}_reverberation.wav')
            self.write(summation, 'summation', f'{recipe["filepath"]}_summation.wav')
            self.write(noise, 'noise', f'{recipe["filepath"]}_noise.wav')

    def generate_samples(self, recipe):
        return recipe.apply(self.generate_sample, axis=1)

    def get_reflections(self, count):
        rng = self.rng
        amplitudes = [rng.get_amplitude() for _ in range(count)]
        delays = [rng.get_delay() for _ in range(count)]
        zeniths = [rng.get_zenith() for _ in range(count)]
        azimuths = [rng.get_azimuth(zenith=zeniths[i]) for i in range(count)]
        return zeniths, azimuths, amplitudes, delays

    def write(self, file, directory, filename):
        from pathlib import Path
        import os

        path = Path(self.output_directory) / directory
        if not os.path.isdir(path):
            os.mkdir(path)

        path = path / filename
        if not os.path.isfile(path):
            print(f'\tWriting file: {filename}')
            file_format = path.suffix.strip('.')
            if file_format == 'wav':
                if self.fs != file.frame_rate:
                    file = file.set_frame_rate(self.fs)
                file.export(path, format=file_format)
            if file_format == 'png':
                file.savefig(path, format=file_format)
                file.figure().clear()
                file.close()
                file.cla()
                file.clf()

        return path
