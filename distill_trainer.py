from transformers import Seq2SeqTrainer
import torch
from torch.nn import MSELoss



class QTrainer(Seq2SeqTrainer):
    
    def __init__(self, 
                 teacher_model=None, 
                 task_weight = 1,
                 logits_weight = 1,
                 hid_weight = 1,
                 distill_enc_mapping = None,
                 distill_dec_mapping = None,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.cuda()
        self.task_weight = task_weight
        self.logits_weight = logits_weight
        self.hid_weight = hid_weight

        self.distill_enc_mapping = distill_enc_mapping
        self.distill_dec_mapping = distill_dec_mapping

        if self.teacher_model is not None:
            self.teacher_model.eval()
    
    def compute_loss(self, model, inputs, return_outputs=False):

        loss_mse = MSELoss()
        student_outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        task_loss = student_outputs.loss

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs, output_attentions=True, output_hidden_states=True)
        
        logits_loss = loss_mse(student_outputs.logits, teacher_outputs.logits)

        enc_att_loss, enc_hid_last_loss, enc_hid_loss, crs_att_loss, dec_att_loss, dec_hid_loss  = 0, 0, 0, 0, 0, 0

        for i, student_att in enumerate(student_outputs.encoder_attentions):
            mapped_idx = self.distill_enc_mapping[i]
            teacher_att = teacher_outputs.encoder_attentions[mapped_idx]
            enc_att_loss += loss_mse(student_att, teacher_att)

        for student_hs, teacher_hs in zip(student_outputs.encoder_last_hidden_state,
                                            teacher_outputs.encoder_last_hidden_state):
            enc_hid_last_loss += loss_mse(student_hs, teacher_hs)

        for i, student_hs in enumerate(student_outputs.encoder_hidden_states):
            if i == 0:
                mapped_idx = 0
            else:
                mapped_idx = self.distill_enc_mapping[i - 1] + 1

            teacher_hs = teacher_outputs.encoder_hidden_states[mapped_idx]
            enc_hid_loss += loss_mse(student_hs, teacher_hs)

        for i, student_att in enumerate(student_outputs.cross_attentions):
            mapped_idx = self.distill_dec_mapping[i]
            teacher_att = teacher_outputs.cross_attentions[mapped_idx]
            crs_att_loss += loss_mse(student_att, teacher_att)

        for i, student_att in enumerate(student_outputs.decoder_attentions):
            mapped_idx = self.distill_dec_mapping[i]
            teacher_att = teacher_outputs.decoder_attentions[mapped_idx]
            dec_att_loss += loss_mse(student_att, teacher_att)

        for i, student_hs in enumerate(student_outputs.decoder_hidden_states):
            if i == 0:
                mapped_idx = 0
            else:
                mapped_idx = self.distill_dec_mapping[i - 1] + 1

            teacher_hs = teacher_outputs.decoder_hidden_states[mapped_idx]
            dec_hid_loss += loss_mse(student_hs, teacher_hs)
        total_loss = self.task_weight * task_loss + \
                        self.logits_weight * logits_loss + \
                        self.hid_weight * (enc_att_loss + dec_att_loss + crs_att_loss + enc_hid_loss + enc_hid_last_loss + dec_hid_loss)
        
        return (total_loss, student_outputs) if return_outputs else total_loss

