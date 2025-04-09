from xml_parser import xml_ref_parser
import argparse
from crossref_matcher import crossref_matcher

def main():
    arg_parser = argparse.ArgumentParser(description='Process XML files.')
    arg_parser.add_argument('path', type=str, help='Path to the XML files directory')
    # arg_parser.add_argument(
    #     'output_path', type=str, help='Path to save the output CSV file'
    # )

    args = arg_parser.parse_args()

    ref_parser = xml_ref_parser(args.path)

    ref_parser.process_xml_directory()
    ref_parser.format()
    matcher = crossref_matcher(ref_parser.df)
    matcher.process_parsed_data()


if __name__ == '__main__':
    main()
