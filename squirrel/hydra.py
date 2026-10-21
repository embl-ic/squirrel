
def get_acquisition_transforms():

    # ----------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Determines transformations based on tiff metadata',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dirpath', type=str,
                        help='Directory containing the tif slices')
    parser.add_argument('out_filepath', type=str,
                        help='Output json which will contain the transformations.')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    dirpath = args.dirpath
    out_filepath = args.out_filepath
    verbose = args.verbose

    from squirrel.workflows.hydra import get_acquisition_transforms_workflow

    get_acquisition_transforms_workflow(
        dirpath,
        out_filepath=out_filepath,
        verbose=verbose
    )


def extract_tif_metadata():
    """Export complete or selected Hydra TIFF metadata to YAML."""
    import argparse

    from squirrel.library.hydra import (
        HYDRA_METADATA_FIELDS,
        HYDRA_METADATA_PRESETS,
        extract_hydra_metadata,
        read_hydra_metadata,
        save_hydra_metadata,
    )

    parser = argparse.ArgumentParser(
        description='Export Hydra TIFF metadata as YAML.'
    )
    parser.add_argument('input_tif', help='Input Hydra TIFF file')
    parser.add_argument(
        '-o', '--out_filepath',
        help='Save YAML to this file (default: print to console)'
    )
    parser.add_argument(
        '--preset', choices=sorted(HYDRA_METADATA_PRESETS),
        help='Export the named field preset'
    )
    parser.add_argument(
        '--fields', nargs='+', metavar='FIELD',
        help='Additional named metadata fields: ' +
             ', '.join(sorted(HYDRA_METADATA_FIELDS))
    )
    args = parser.parse_args()

    if args.preset or args.fields:
        metadata = extract_hydra_metadata(
            args.input_tif, preset=args.preset, fields=args.fields
        )
    else:
        metadata = read_hydra_metadata(args.input_tif)

    if args.out_filepath is not None:
        save_hydra_metadata(metadata, args.out_filepath)
    else:
        import yaml
        print(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), end='')
