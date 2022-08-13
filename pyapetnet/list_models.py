import os
import argparse
from glob import glob


def main():
    """ list available pre-trained pyapetnet models """

    parser = argparse.ArgumentParser(
        description='list available trained models for pyapetnet')
    parser.add_argument(
        '--model_path',
        help='absolute path of directory containing trained models',
        default=None)
    args = parser.parse_args()

    # parse input parameters
    import pyapetnet
    model_path = args.model_path

    if model_path is None:
        model_path = os.path.join(os.path.dirname(pyapetnet.__file__),
                                  'trained_models')

    cfg_files = sorted(glob(os.path.join(model_path, '*', 'config.json')))

    print(f'\nModel path: {model_path}')
    print('\nAvailable models')
    print('----------------')

    for i, cfg_file in enumerate(cfg_files):
        print(f'{os.path.basename(os.path.dirname(cfg_file))}')

    print(
        f'\nFor details about the models, read \n{os.path.join(model_path,"model_description.md")}\nor the look at the config.json files in the model directories'
    )


if __name__ == '__main__':
    main()