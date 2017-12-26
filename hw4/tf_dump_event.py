import sys
import os
import tensorflow as tf

def dump_all_imgs(events_path, dump_dir):
    for e in tf.train.summary_iterator(events_path):
        for v in e.summary.value:
            if hasattr(v, 'image'):
                img_name = '{}.jpg'.format(e.step)
                img_path = os.sep.join([dump_dir, v.tag, img_name])
                output_dir = os.path.dirname(img_path)
                if len(output_dir) > 0 and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(img_path, 'wb') as f:
                    _ = f.write(v.image.encoded_image_string)

if __name__ == '__main__':
    events_path = sys.argv[1] if len(sys.argv) > 1 else 'events.out.tfevents.1513832614.PCD-DIY-ASUS'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'dump/'
    print('event file: {}'.format(events_path))
    print('output dir: {}'.format(output_dir))
    dump_all_imgs(events_path, output_dir)
    print('done')