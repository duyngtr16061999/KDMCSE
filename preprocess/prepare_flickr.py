import xml.etree.ElementTree as ET

import argparse
import os.path as osp
import tqdm
import random

from extract_visn_feature import ResnetFeatureExtractor


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append({'first_word_index': index,
                                             'phrase': phrase,
                                             'phrase_id': p_id,
                                             'phrase_type': p_type})

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes': {}, 'scene': [], 'nobox': []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info


def get_flickr_caps(split, flickr_entity_dir, output_dir):
    # test/val: 1000*5 captions      train: 29783*5=148915 captions
    caption_path = osp.join(flickr_entity_dir, 'Sentences')

    id_path = osp.join(flickr_entity_dir, '%s.txt'%split)

    print('Start sampling captions ...')
    img_ids = open(id_path, 'r').readlines()
    img_ids = [id.strip() for id in img_ids]
    sent_paths = [osp.join(caption_path, id + '.txt') for id in img_ids]
    captions = []

    for i, sent_path in enumerate(tqdm.tqdm(sent_paths)):
        annotations = get_sentence_data(sent_path)
        sentences = [a['sentence'] for a in annotations]
        captions.extend(random.sample(sentences, 1))

    # Save captions
    curr_paths_fname = osp.join(output_dir, 'flickr_random_captions.txt')
    print("Save captions to ", curr_paths_fname)
    with open(curr_paths_fname, 'w') as f:
        f.write("\n".join(captions))

    return img_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flickr_entities_dir', type=str, default='the_flickr30k_entities_path')
    parser.add_argument('--flickr_images_dir', type=str, default='the_flickr30k_images_path')
    parser.add_argument('--output_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    # get random captions
    img_ids = get_flickr_caps(split='train',
                              flickr_entity_dir=args.flickr_entities_dir,
                              output_dir=args.output_dir)

    # extract image features
    extractor = ResnetFeatureExtractor(args.flickr_images_dir, args.output_dir, args.batch_size)
    extractor.extract_vision_features(dataname='flickr',
                                      img_ids=img_ids)
