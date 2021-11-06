
dataset_path = '/opt/ml/segmentation/input/data'

test_path = dataset_path + '/test.json'

test_transform = A.Compose([
                           ToTensorV2()
                           ])
