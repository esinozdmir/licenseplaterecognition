import os
import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_csv(path):
    xml_list = []
    for xml_file in os.listdir(path):
        if xml_file.endswith('.xml'):  # Sadece XML dosyalarını oku
            tree = ET.parse(os.path.join(path, xml_file))
            root = tree.getroot()
            for member in root.findall('object'):  # Her bir nesne için (plaka)
                value = (
                    root.find('filename').text,  # Görsel dosya adı
                    int(root.find('size/width').text),  # Görsel genişliği
                    int(root.find('size/height').text),  # Görsel yüksekliği
                    member.find('name').text,  # Nesne sınıfı (licence)
                    int(member.find('bndbox/xmin').text),  # Bndbox xmin
                    int(member.find('bndbox/ymin').text),  # Bndbox ymin
                    int(member.find('bndbox/xmax').text),  # Bndbox xmax
                    int(member.find('bndbox/ymax').text)   # Bndbox ymax
                )
                xml_list.append(value)
    # DataFrame oluştur
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

if __name__ == "__main__":
    annotations_path = './veri_seti/annotations'  # XML dosyalarının bulunduğu klasör
    output_csv = 'labels.csv'  # Çıkış dosyası adı
    xml_df = xml_to_csv(annotations_path)
    xml_df.to_csv(output_csv, index=False)
    print(f"CSV dosyası başarıyla oluşturuldu: {output_csv}")
