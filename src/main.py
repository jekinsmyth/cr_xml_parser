from xml_parser import xml_ref_parser
import argparse


def main():
    arg_parser = argparse.ArgumentParser(description='Process XML files.')
    arg_parser.add_argument('path', type=str, help='Path to the XML files directory')
    arg_parser.add_argument(
        'output_path', type=str, help='Path to save the output CSV file'
    )

    args = arg_parser.parse_args()

    ref_parser = xml_ref_parser(args.path)

    ref_parser.process_xml_directory()
    ref_parser.format()
    ref_parser.save_to_csv(args.output_path)


if __name__ == '__main__':
    main()
