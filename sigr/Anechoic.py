from torch.utils.data import Dataset


class Anechoic(Dataset):

    def __init__(self, root, ttv, download=False, transform=None, target_transform=None, columns=None,
                 output_path=None):
        from pathlib import Path
        import os

        ttvs = ['train', 'test', 'validate']
        assert ttv in ttvs, f'Acceptable values for ttv are {", ".join(ttvs)}'

        self.ttv = ttv
        self.transform = transform
        self.target_transform = target_transform
        self.root = Path(root).__str__()
        self.data_path = (Path(self.root) / self.ttv).__str__()
        self.label_path = f'{self.data_path}_recipe'
        self.output_path = output_path

        if download:
            self.download()
        else:
            assert os.path.isdir(self.root), f'Root directory {self.root} must exist if download=False'
            assert os.path.isdir(self.data_path), f'Data directory {self.data_path} must exist if download=False'
            assert os.path.isdir(self.label_path), f'Label directory {self.label_path} must exist if download=False'

        self.labels = self.set_labels(columns)

    def download(self):
        from pathlib import Path
        import requests
        import zipfile
        import io
        import shutil
        import os

        if not os.path.isdir(self.root):
            os.mkdir(self.root)

        _download_url = 'https://reflections.speakeasy.services'

        print(f'Downloading dataset at {_download_url}/{self.ttv}.zip')
        r = requests.get(f'{_download_url}/{self.ttv}.zip', stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print(f'Finished downloading')

        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.label_path):
            os.mkdir(self.label_path)

        print('Extracting dataset')
        for f in z.namelist():
            filename = Path(f).name
            if not filename:
                continue
            source = z.open(f)
            if filename.endswith('.zip'):
                target = open((Path(self.root) / filename).__str__(), 'wb')
            else:
                target = open((Path(self.data_path) / filename).__str__(), 'wb')
            print(f'\tExtracting file: {filename}')
            with source, target:
                shutil.copyfileobj(source, target)

        assert os.path.isfile(f'{self.label_path}.zip'), f'{self.label_path}.zip missing'
        z = zipfile.ZipFile(f'{self.label_path}.zip')
        z.extractall(self.label_path)

    def set_labels(self, columns):
        from data_loader import read_recipe

        if columns is not None:
            if type(columns) is not None:
                columns = [columns]
            if 'filepath' not in columns:
                columns.append('filepath')
            return read_recipe(self.label_path)[columns]

        return read_recipe(self.label_path)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        from pydub import AudioSegment
        from pathlib import Path
        from utils import audiosegment_to_array

        labels = self.labels.iloc[item]
        audio = AudioSegment.from_wav((Path(self.data_path) / f"{labels['filepath']}.wav").__str__())

        if self.transform:
            audio = self.transform(audio)
        else:
            audio = audiosegment_to_array(audio)
        if self.target_transform:
            labels = self.target_transform(labels)

        return audio, labels

    def play_sample(self, item):
        from pathlib import Path
        from pydub import AudioSegment
        from utils import play_audio
        from IPython.display import display
        import os

        filepath = f'{(Path(self.data_path) / self.labels.iloc[item]["filepath"]).__str__()}.wav'
        assert os.path.isfile(filepath), f'{filepath} does not exist'
        audio = AudioSegment.from_wav(filepath)
        return display(play_audio(audio))


def get_ttv(root, download=False, transform=None, target_transform=None, columns=None, batch_size=60):
    from torch.utils.data import DataLoader

    train = DataLoader(
        Anechoic(root, 'train', download=download, transform=transform, target_transform=target_transform,
                 columns=columns), batch_size=batch_size, shuffle=True)
    test = DataLoader(Anechoic(root, 'test', download=download, transform=transform, target_transform=target_transform,
                               columns=columns), batch_size=batch_size, shuffle=False)
    validate = DataLoader(
        Anechoic(root, 'validate', download=download, transform=transform, target_transform=target_transform,
                 columns=columns), batch_size=batch_size, shuffle=True)

    return train, test, validate
