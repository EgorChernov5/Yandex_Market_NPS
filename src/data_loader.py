from torch.utils.data import Dataset, DataLoader


class goodsDataset(Dataset):
    LABELS = {
        "clean_photo_good_background": 0,
        "other_infographics": 1,
        "good_infographics": 2,
        "clean_photo_other_background": 3,
        "bad_infographics": 4,
        "clean_photo_bad_background": 5,
        "clean_photo_image_background": 6
    }

    def __init__(self, df, transform=None):
        """
        Arguments:
            df : pandas DataFrame.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = df
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_url = "https://" + self.data_frame[["pic_url"]].iloc[idx][0]
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = response.content
            image = Image.open(BytesIO(image_data))
        else:
            return "Не удалось загрузить изображение"

        if self.transform:
            image = self.transform(image)

        # sample = {'image': image, 'label': goodsDataset.LABELS[self.data_frame[["verdict"]].iloc[idx][0]]}
        sample = [image, goodsDataset.LABELS[self.data_frame[["verdict"]].iloc[idx][0]]]
        return sample


def load_data(df, height=300, weight=300, batch_size=4, num_workers=0):
    transform = t.Compose([
        t.Resize((height, weight)),
        t.ToTensor()
    ])
    goods_dataset = goodsDataset(df=df, transform=transform)
    dataloader = DataLoader(goods_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    return dataloader
