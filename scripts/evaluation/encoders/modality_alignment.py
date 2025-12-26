"""
Modality alignment tests
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn.functional as F

from core.embeddings.biomedclip import BioMedCLIPEncoder
from core.kb.image_loader import ImageLoader


class ModalityAlignmentTester:
    """Tests alignment between modalities"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.encoder = BioMedCLIPEncoder(device=device)
        self.image_loader = ImageLoader()
    
    def test(self):
        """Run alignment tests"""
        test_image_path = "data/images/edema_Image_1.jpg"
        if not Path(test_image_path).exists():
            return {"status": "test_image_not_found"}
        
        return {
            "related_similarity": self._test_related_pairs(),
            "unrelated_similarity": self._test_unrelated_pairs()
        }
    
    def _test_related_pairs(self):
        """Test similarity of related image-text pairs"""
        # Test with actual data if available
        pairs = [
            ("data/images/edema_Image_1.jpg", 
             "Hi doctor my son is four years old....from one year he has a snorring habit while sleeping at night...the sound used to increase when he have any cold and cough problems...we used to ignore it as normal..recently we planned a minor surgery for him for phimosis..At the time when doctor injected anasthesia he suddenly developed bronchspasm and an acute laryngeal .Please see the current condition below .doctor then called us for further check up for finding the reason for this...he observed that he is breething through his nose while sleeping..then doctor suggested for an digital Xray of his upper airways..then they found mild enlargement of adenoid gland (type 2)..But our Doctor is still doubting he has some problem related to larynx..and recommended us to visit ENT specialist..actually we are totally worried now..we dont the actual reason for our child s negative reaction to anasthesia..Doctor give us a hint that he may need an surgery"),
            ("data/images/cyanosis_Image_1.jpg",
             "Hi, my daughter is 9 month old. on a regular 7day she looks beautifully healthy but then out of no where she turn pale, and gets real dark circles under her eyes, she becomes tired like an nonresponsive but awake. Cold to the touch and temperature drops close to 1degree or a little more, blueish purpleish lips, and then, just as sudden as it came it goes away after a few minutes. she has been sleepin well and on some occassions a little more than normal. eats great, drinks milk and plenty of water. Im beginning to get worred, please see the image of the affected area below .")
        ]
        
        similarities = []
        for img_path, text in pairs:
            if Path(img_path).exists():
                img = self.image_loader.load(img_path)
                img_emb = self.encoder.encode_image(img)
                txt_emb = self.encoder.encode_text(text)
                
                sim = F.cosine_similarity(
                    img_emb, txt_emb, dim=0
                ).item()
                similarities.append(sim)
        
        if similarities:
            return {
                "mean": sum(similarities) / len(similarities),
                "values": similarities
            }
        return {"status": "no_test_pairs_available"}
    
    def _test_unrelated_pairs(self):
        """Test similarity of unrelated pairs"""
        # Cross-category comparisons
        pairs = [
            ("data/images/edema_Image_1.jpg",
             "bluish discoloration observed"),
            ("data/images/cyanosis_Image_1.jpg",
             "swelling noted around joint")
        ]
        
        similarities = []
        for img_path, text in pairs:
            if Path(img_path).exists():
                img = self.image_loader.load(img_path)
                img_emb = self.encoder.encode_image(img)
                txt_emb = self.encoder.encode_text(text)
                
                sim = F.cosine_similarity(
                    img_emb, txt_emb, dim=0
                ).item()
                similarities.append(sim)
        
        if similarities:
            return {
                "mean": sum(similarities) / len(similarities),
                "values": similarities
            }
        return {"status": "no_test_pairs_available"}