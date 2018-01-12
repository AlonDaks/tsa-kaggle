import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tsa.utils import data_path as dp


def parse_bbox_xml(file_name):
  tree = ET.parse(file_name)
  bboxes = []
  for _object in tree.getroot().iter('object'):
    bndbox = _object.find('bndbox')
    bboxes.append([
      int(bndbox.find('xmin').text),
      int(bndbox.find('ymin').text),
      int(bndbox.find('xmax').text),
      int(bndbox.find('ymax').text)
    ])
  return bboxes


def write_xml(folder, filename, path, objects):
  def helper(out_path):
    annotation = ET.Element('annotation')

    folder_elem = ET.Element('folder')
    folder_elem.text = folder
    filename_elem = ET.Element('filename')
    filename_elem.text = filename
    path_elem = ET.Element('path')
    path_elem.text = path
    source = ET.Element('source')
    database = ET.Element('database')
    database.text = 'Unknown'
    source.append(database)
    size = ET.Element('size')
    width = ET.Element('width')
    width.text = '512'
    height = ET.Element('height')
    height.text = '660'
    depth = ET.Element('depth')
    depth.text = '1'
    size.extend([width, height, depth])
    segmented = ET.Element('segmented')
    segmented.text = '0'

    annotation.extend(
      [folder_elem, filename_elem, path_elem, source, size, segmented])

    for bbox in objects:
      object_element = ET.Element('object')
      name = ET.Element('name')
      name.text = 'threat'
      pose = ET.Element('pose')
      pose.text = 'Unspecified'
      truncated = ET.Element('truncated')
      truncated.text = '0'
      difficult = ET.Element('difficult')
      difficult.text = '0'
      bndbox = ET.Element('bndbox')
      xmin = ET.Element('xmin')
      xmin.text = str(bbox[0])
      ymin = ET.Element('ymin')
      ymin.text = str(bbox[1])
      xmax = ET.Element('xmax')
      xmax.text = str(bbox[2])
      ymax = ET.Element('ymax')
      ymax.text = str(bbox[3])
      bndbox.extend([xmin, ymin, xmax, ymax])
      object_element.extend([name, pose, truncated, difficult, bndbox])
      annotation.append(object_element)

    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(
      indent='    ')
    xmlstr = xmlstr.replace('<?xml version="1.0" ?>\n', '')
    with open(out_path, "w") as f:
      f.write(xmlstr)

  return helper


def merge_xml(bbox_dir, zones, output_dir):
  bbox_map = dict()
  for zone in zones:
    zone_dir = '{0}/zone_{1}'.format(bbox_dir, zone)
    for path in os.listdir(zone_dir):
      if path not in bbox_map:
        bbox_map[path] = parse_bbox_xml(zone_dir + '/' + path)
      else:
        bbox_map[path] += parse_bbox_xml(zone_dir + '/' + path)
  for path, objects in bbox_map.items():
    if not os.path.exists(output_dir + '/' + path):
      write_xml(None, None, None, objects)(output_dir + '/' + path)


if __name__ == '__main__':
  merge_xml(dp.REPO_HOME_PATH + '/data/bbox/aps_threats/',
            [13, 14, 15, 16], dp.REPO_HOME_PATH + '/data/bbox/aps_merged_threats_calf/')
  merge_xml(dp.REPO_HOME_PATH + '/data/bbox/aps_threats/',
            [8, 9, 10, 11, 12], dp.REPO_HOME_PATH + '/data/bbox/aps_merged_threats_thigh/')
  merge_xml(dp.REPO_HOME_PATH + '/data/bbox/aps_threats/',
            [1, 2, 3, 4], dp.REPO_HOME_PATH + '/data/bbox/aps_merged_threats_arm/')