#!/usr/bin/env python

from __future__ import print_function
from glob import glob 
import argparse
import xml.etree.ElementTree as et
from collections import defaultdict 
from os.path import join 
import codecs 
from traceback import format_exc
from extract_conll_framenet_roles import save_frame2role2lu


sample_xml = """
<sentence corpID="246" docID="25616" sentNo="1" paragNo="1" aPos="0" ID="4164976">
    <text>Forest fires continue to rage in Spain</text>
    <annotationSet cDate="07/17/2014 11:43:11 PDT Thu" status="UNANN" ID="6669146">
        <layer rank="1" name="PENN">
            <label end="5" start="0" name="nn"/>
            <label end="11" start="7" name="nns"/>
            <label end="20" start="13" name="VVP"/>
            <label end="23" start="22" name="to"/>
            <label end="28" start="25" name="VV"/>
            <label end="31" start="30" name="in"/>
            <label end="37" start="33" name="NP"/>
        </layer>
        <layer rank="1" name="NER">
            <label end="37" start="33" name="location"/>
        </layer>
        <layer rank="1" name="WSL">
            <label end="23" start="22" name="NT"/>
            <label end="31" start="30" name="NT"/>
            <label end="37" start="33" name="NT"/>
        </layer>
    </annotationSet>
    <annotationSet cDate="07/17/2014 01:15:28 PDT Thu" luID="17455" luName="fire.n" frameID="2824"             frameName="Fire_burning" status="MANUAL" ID="6669245">
        <layer rank="1" name="Target">
            <label cBy="WMcQ" end="11" start="7" name="Target"/>
        </layer>
        <layer rank="1" name="FE">
            <label cBy="WMcQ" feID="16020" bgColor="800080" fgColor="FFFFFF" end="5" start="0" name="Fuel"/>
            <label cBy="WMcQ" feID="16018" bgColor="0000FF" fgColor="FFFFFF" end="11" start="7" name="Fire"/>
        </layer>
        <layer rank="1" name="GF">
            <label end="5" start="0" name="Dep"/>
        </layer>
        <layer rank="1" name="PT">
            <label end="5" start="0" name="N"/>
        </layer>
        <layer rank="1" name="Other"/>
        <layer rank="1" name="Sent"/>
        <layer rank="1" name="Noun"/>
    </annotationSet>
</sentence>
"""


def extact_frame2role2lu(xml_dir, output_dir, verbose = False):
    """ Count frame - role - lu frequencies in the sentences """
    
    xml_fpaths = join(xml_dir, "*xml")
    
    frame2role2lu = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for i, text_fpath in enumerate(glob(xml_fpaths)):
        if verbose: print(text_fpath)

        tree = et.parse(text_fpath)
        root = tree.getroot()
        for child in root:
            if child.tag.endswith("sentence"):
                text = ""
                for gchild in child:
                    if gchild.tag.endswith("text"):
                        text = gchild.text
                        if verbose: print("="*50, "\n", text)
                    elif gchild.tag.endswith("annotationSet") and "frameID" in gchild.attrib:
                        fee = gchild.attrib["luName"]
                        frame_id = gchild.attrib["frameID"]
                        frame_name = gchild.attrib["frameName"]
                        frame2role2lu[frame_name]["FEE"][fee] += 1
                        if verbose: print("\n>>>",fee, frame_id, frame_name)
                        for ggchild in gchild:
                            if ggchild.tag.endswith("layer") and ggchild.attrib["name"] == "FE":
                                for gggchild in ggchild:
                                    if gggchild.tag.endswith("label") and "end" in gggchild.attrib:
                                        role_beg = int(gggchild.attrib["start"])
                                        role_end = int(gggchild.attrib["end"]) + 1
                                        role_name = gggchild.attrib["name"]
                                        role_lu = text[role_beg:role_end]
                                        if verbose: print(role_name, ":", role_lu)
                                        frame2role2lu[frame_name][role_name][role_lu] += 1
    return frame2role2lu
                         

def run(xml_dir, output_fpath):
    """ Gets the path to the framenet root directory and output the csv file with role lus. """

    frame2role2lu = extact_frame2role2lu(xml_dir, output_fpath)
    save_frame2role2lu(frame2role2lu, output_fpath)


def main():
    parser = argparse.ArgumentParser(description='Extracts an evaluation dataset for role induction'
            'from the xml framenet files.')
    parser.add_argument('xml_dir', help='Directory with the xml files of framenet.')
    parser.add_argument('output_fpath', help='Output directory.')
    args = parser.parse_args()
    print("Input: ", args.xml_dir)
    print("Output: ", args.output_fpath)

    run(args.xml_dir, args.output_fpath)


if __name__ == '__main__':
    main()

