from lxml import etree as ET
import os
import pandas as pd


class xml_ref_parser:
    def __init__(self, path):
        self.path = path
        self.path_list = os.listdir(path)
        self.metadata = []
        self.df = pd.DataFrame()
        self.reference_list = []

    def process_xml_directory(self):
        """
        Process all XML files in the specified directory.
        """
        for file in self.path_list:
            if file.endswith('.xml'):
                self.parse_pubmed_xml(file)

    def parse_pubmed_xml(self, file):
        """
        Parse XML files in the specified directory and extract metadata.
        """

        try:
            tree = ET.parse(os.path.join(self.path, file))
            treeroot = tree.getroot()

            references_section = treeroot.xpath(
                "//*[local-name()='title' and text()='References']/following-sibling::*[local-name()='ref' and @id]"
            )

            for ref in references_section:
                authors = ref.xpath(
                    ".//*[local-name()='person-group']/*[local-name()='name']"
                )
                author_list = []
                for author in authors:
                    surname = author.xpath("./*[local-name()='surname']/text()")
                    given_names = author.xpath("./*[local-name()='given-names']/text()")
                    full_name = (
                        f'{" ".join(given_names)} {" ".join(surname)}'
                        if surname and given_names
                        else 'Unknown Author'
                    )
                    author_list.append(full_name)

                etal = ref.xpath(
                    ".//*[local-name()='person-group']/*[local-name()='etal']"
                )
                if etal:
                    author_list.append('et al.')

                article_title_element = ref.xpath(".//*[local-name()='article-title']")
                if article_title_element:
                    article_title = ET.tostring(
                        article_title_element[0], encoding='unicode', method='text'
                    ).strip()
                else:
                    article_title = None
                source = ref.xpath(".//*[local-name()='source']/text()")
                source = source[0] if source else None

                year = ref.xpath(".//*[local-name()='year']/text()")
                year = year[0] if year else None

                volume = ref.xpath(".//*[local-name()='volume']/text()")
                volume = volume[0] if volume else None

                fpage = ref.xpath(".//*[local-name()='fpage']/text()")
                fpage = fpage[0] if fpage else None

                lpage = ref.xpath(".//*[local-name()='lpage']/text()")
                lpage = lpage[0] if lpage else None

                pub_id_pmid = ref.xpath(
                    ".//*[local-name()='pub-id' and @pub-id-type='pmid']/text()"
                )
                pub_id_pmid = pub_id_pmid[0] if pub_id_pmid else None

                pub_id_doi = ref.xpath(
                    ".//*[local-name()='pub-id' and @pub-id-type='doi']/text()"
                )
                pub_id_doi = pub_id_doi[0] if pub_id_doi else None

                file = file.replace('.xml', '')

                self.metadata.append(
                    {
                        'ref_id': file,
                        'article_title': article_title,
                        'authors': author_list,
                        'source': source,
                        'year': year,
                        'volume': volume,
                        'fpage': fpage,
                        'lpage': lpage,
                        'pmid': pub_id_pmid,
                        'doi': pub_id_doi,
                    }
                )
        except Exception:
            pass

        self.df = pd.DataFrame(self.metadata)

    def format(self):
        """
        Takes metadata and formats it in Harvard style.
        """
        self.df['formatted_reference'] = self.df.apply(
            lambda row: f'{", ".join(row["authors"])}, {row["year"]}, {row["article_title"]}, {row["source"]}, {row["volume"]}, {row["fpage"]}-{row["lpage"]}, {row["pmid"]}, {row["doi"]}',
            axis=1,
        )

    def get_references_list(self):
        """
        Returns the formatted references list.
        """
        self.reference_list = self.df['formatted_reference'].tolist()
        return self.reference_list

    def save_to_csv(self, output_path):
        """
        Save the DataFrame to a CSV file.
        """
        self.df.to_csv(output_path, index=False)
