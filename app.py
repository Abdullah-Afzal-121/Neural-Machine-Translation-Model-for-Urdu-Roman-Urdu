import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import re
import unicodedata
from collections import namedtuple
import random
import os

# Set page config first
st.set_page_config(
    page_title="Urdu to Roman Urdu Translator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# 1. Model Architecture (Same as your training code)
# =====================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=1)
        self.rnn = nn.LSTM(emb_dim, enc_hidden_dim, num_layers=n_layers,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0.0, 
                          batch_first=True)
        
        self.fc_hidden = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.fc_cell = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len=None):
        embedded = self.dropout(self.embedding(src))
        
        if src_len is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        if src_len is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_cat = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        
        cell_fwd = cell[-2, :, :]
        cell_bwd = cell[-1, :, :]
        cell_cat = torch.cat((cell_fwd, cell_bwd), dim=1)
        
        hidden_proj = torch.tanh(self.fc_hidden(hidden_cat))
        cell_proj = torch.tanh(self.fc_cell(cell_cat))
        
        hidden_proj = hidden_proj.unsqueeze(0).repeat(self.n_layers, 1, 1)
        cell_proj = cell_proj.unsqueeze(0).repeat(self.n_layers, 1, 1)
        
        return outputs, hidden_proj, cell_proj

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=1)
        self.rnn = nn.LSTM(emb_dim + enc_hidden_dim * 2, dec_hidden_dim, num_layers=n_layers,
                          dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        
        self.fc_out = nn.Linear(emb_dim + enc_hidden_dim * 2 + dec_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden[-1], encoder_outputs, mask)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        
        return prediction, hidden, cell, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src != self.src_pad_idx)
        return mask
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        mask = self.create_mask(src)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1], device=self.device)
        
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell, attention = self.decoder(input, hidden, cell, encoder_outputs, mask)
            
            outputs[:, t] = output
            attentions[:, t] = attention
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            input = trg[:, t] if teacher_force else top1
            
        return outputs, attentions

def create_model(src_vocab_size, trg_vocab_size, device, 
                emb_dim=256, enc_hidden_dim=256, dec_hidden_dim=256, 
                n_layers=2, dropout=0.3, src_pad_idx=1):
    
    attention = Attention(enc_hidden_dim, dec_hidden_dim)
    encoder = Encoder(src_vocab_size, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers, dropout)
    decoder = Decoder(trg_vocab_size, emb_dim, enc_hidden_dim, dec_hidden_dim, n_layers, dropout, attention)
    
    model = Seq2Seq(encoder, decoder, src_pad_idx, device).to(device)
    
    return model

# =====================================================
# 2. Translation Functions
# =====================================================
BeamNode = namedtuple('BeamNode', ['tokens', 'log_prob', 'hidden', 'cell', 'finished'])

def beam_search_translate(model, sentence, sp_src, sp_trg, beam_size=5, max_len=50, device='cpu'):
    """Beam search translation"""
    model.eval()
    
    try:
        tokens = [sp_src.bos_id()] + sp_src.encode(sentence.strip(), out_type=int) + [sp_src.eos_id()]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mask = model.create_mask(src_tensor)
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
            
            initial_token = sp_trg.bos_id()
            beams = [BeamNode(
                tokens=[initial_token],
                log_prob=0.0,
                hidden=hidden,
                cell=cell,
                finished=False
            )]
            
            finished_beams = []
            
            for step in range(max_len):
                if not beams:
                    break
                    
                candidates = []
                
                for beam in beams:
                    if beam.finished:
                        finished_beams.append(beam)
                        continue
                    
                    input_token = torch.LongTensor([beam.tokens[-1]]).to(device)
                    output, new_hidden, new_cell, _ = model.decoder(
                        input_token, beam.hidden, beam.cell, encoder_outputs, mask
                    )
                    
                    log_probs = F.log_softmax(output, dim=-1)
                    top_log_probs, top_indices = torch.topk(log_probs, min(beam_size, log_probs.size(-1)))
                    
                    for i in range(min(beam_size, log_probs.size(-1))):
                        token_id = top_indices[0, i].item()
                        token_log_prob = top_log_probs[0, i].item()
                        
                        new_tokens = beam.tokens + [token_id]
                        new_log_prob = beam.log_prob + token_log_prob
                        
                        is_finished = (token_id == sp_trg.eos_id())
                        
                        candidates.append(BeamNode(
                            tokens=new_tokens,
                            log_prob=new_log_prob,
                            hidden=new_hidden.clone(),
                            cell=new_cell.clone(),
                            finished=is_finished
                        ))
                
                candidates.sort(key=lambda x: x.log_prob / len(x.tokens), reverse=True)
                beams = [beam for beam in candidates[:beam_size] if not beam.finished]
                
                finished_beams.extend([beam for beam in candidates if beam.finished])
            
            finished_beams.extend(beams)
            
            if not finished_beams:
                return "Translation failed"
            
            best_beam = max(finished_beams, key=lambda x: x.log_prob / len(x.tokens))
            
            decoded_tokens = [t for t in best_beam.tokens[1:] if t != sp_trg.eos_id()]
            return sp_trg.decode(decoded_tokens).strip()
            
    except Exception as e:
        st.error(f"Error in beam search translation: {str(e)}")
        return "Translation error"

def translate_sentence_greedy(sentence, model, sp_src, sp_trg, max_len=50, device='cpu'):
    """Greedy search translation"""
    model.eval()
    
    try:
        tokens = [sp_src.bos_id()] + sp_src.encode(sentence.strip(), out_type=int) + [sp_src.eos_id()]
        src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            mask = model.create_mask(src_tensor)
            encoder_outputs, hidden, cell = model.encoder(src_tensor)

        trg_indexes = [sp_trg.bos_id()]
        
        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
            with torch.no_grad():
                output, hidden, cell, _ = model.decoder(trg_tensor, hidden, cell, encoder_outputs, mask)
                pred_token = output.argmax(1).item()
            
            trg_indexes.append(pred_token)
            
            if pred_token == sp_trg.eos_id():
                break
        
        translated_tokens = [t for t in trg_indexes[1:] if t != sp_trg.eos_id()]
        translated = sp_trg.decode(translated_tokens)
        return translated.strip()
        
    except Exception as e:
        st.error(f"Error in greedy translation: {str(e)}")
        return "Translation error"

# =====================================================
# 3. Text Normalization Functions
# =====================================================
def normalize_urdu(text, remove_tashkeel=True, remove_tatweel=True,
                  normalize_alef=True, normalize_ye=True, normalize_kaf=True,
                  remove_extra_punct=False):
    """Enhanced Urdu normalization"""
    if text is None or not text.strip():
        return ""
    
    s = text.strip()
    
    if remove_tatweel:
        s = re.sub('\u0640+', '', s)
    
    if remove_tashkeel:
        s = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]', '', s)
    
    if normalize_alef:
        s = re.sub('[إأآٱ]', 'ا', s)
    
    if normalize_ye:
        s = s.replace('ي', 'ی')
    
    if normalize_kaf:
        s = s.replace('ك', 'ک')
    
    s = s.replace('ﻻ', 'لا')
    
    s = re.sub(r'\s+', ' ', s).strip()
    
    if remove_extra_punct:
        keep = set(['،', '۔', '؟'])
        s = ''.join(ch for ch in s if (unicodedata.category(ch)[0] != 'P') or (ch in keep))
        s = re.sub(r'\s+', ' ', s).strip()
    
    return s

# =====================================================
# 4. Load Model and Tokenizers
# =====================================================
@st.cache_resource
def load_model_and_tokenizers():
    """Load the trained model and SentencePiece tokenizers"""
    try:
        # Check if files exist
        required_files = [
            "best_seq2seq_model.pt",
            "spm_ur.model", 
            "spm_ro.model"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            st.error(f"❌ Missing required files: {missing_files}")
            return None, None, None, None
        
        # Load SentencePiece models
        sp_ur = spm.SentencePieceProcessor(model_file="spm_ur.model")
        sp_ro = spm.SentencePieceProcessor(model_file="spm_ro.model")
        
        ur_vocab_size = sp_ur.get_piece_size()
        ro_vocab_size = sp_ro.get_piece_size()
        
        # Model hyperparameters (match your training)
        EMB_DIM = 256           
        ENC_HIDDEN_DIM = 384    
        DEC_HIDDEN_DIM = 384    
        N_LAYERS = 2            
        DROPOUT = 0.3
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        model = create_model(
            src_vocab_size=ur_vocab_size,
            trg_vocab_size=ro_vocab_size,
            device=device,
            emb_dim=EMB_DIM,
            enc_hidden_dim=ENC_HIDDEN_DIM,
            dec_hidden_dim=DEC_HIDDEN_DIM,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
            src_pad_idx=sp_ur.pad_id()
        )
        
        # Load trained weights
        checkpoint = torch.load("best_seq2seq_model.pt", map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        st.success(f"✅ Model loaded successfully! Using device: {device}")
        st.info(f"📊 Vocabularies: Urdu={ur_vocab_size:,} tokens, Roman={ro_vocab_size:,} tokens")
        
        return model, sp_ur, sp_ro, device
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {e}")
        st.info("Please ensure these files exist in your directory:")
        st.code("""
        your_folder/
        ├── streamlit_app.py
        ├── best_seq2seq_model.pt
        ├── spm_models/
        │   └── spm_ur.model
        └── smp_models/
            └── smp_ro.model
        """)
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None

# =====================================================
# 5. Initialize session state
# =====================================================
def initialize_session_state():
    """Initialize session state variables"""
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    if 'current_input' not in st.session_state:
        st.session_state.current_input = ""

# =====================================================
# 6. Main Streamlit App
# =====================================================
def main():
    # Header
    st.title("🌍 Urdu ↔ Roman Urdu Translator")
    st.markdown("### Neural Machine Translation with Attention Mechanism")
    
    # Initialize session state
    initialize_session_state()
    
    # Load model
    with st.spinner("🔄 Loading model and tokenizers..."):
        model, sp_ur, sp_ro, device = load_model_and_tokenizers()
    
    if model is None:
        st.stop()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Translation settings
        st.subheader("Translation Options")
        use_beam_search = st.checkbox("🎯 Use Beam Search", value=True, 
                                     help="Better quality translation but slower")
        
        if use_beam_search:
            beam_size = st.slider("Beam Size", 1, 10, 5,
                                help="Higher values = better quality but slower")
        else:
            beam_size = 1
            
        max_length = st.slider("Max Output Length", 20, 100, 50,
                             help="Maximum length of translated text")
        
        # Normalization settings
        st.subheader("Text Normalization")
        normalize_input = st.checkbox("📝 Normalize Input Text", value=True)
        
        if normalize_input:
            with st.expander("Normalization Options"):
                remove_tashkeel = st.checkbox("Remove Tashkeel/Diacritics", value=True)
                remove_tatweel = st.checkbox("Remove Tatweel (Kashida)", value=True)
                normalize_alef = st.checkbox("Normalize Alef variants", value=True)
                normalize_ye = st.checkbox("Normalize Ye variants", value=True)
                normalize_kaf = st.checkbox("Normalize Kaf variants", value=True)
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.translation_history = []
            st.success("History cleared!")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input (Urdu)")
        
        # Text input area
        urdu_input = st.text_area(
            "Enter Urdu text:",
            value=st.session_state.current_input,
            height=250,
            placeholder="یہاں اردو متن لکھیں...\n\nمثال: آج موسم بہت اچھا ہے۔",
            help="Type or paste Urdu text here"
        )
        
        # Update session state
        st.session_state.current_input = urdu_input
        
        # Character and word count
        if urdu_input:
            char_count = len(urdu_input)
            word_count = len(urdu_input.split())
            st.caption(f"📊 Characters: {char_count} | Words: {word_count}")
    
    with col2:
        st.header("🔄 Output (Roman Urdu)")
        
        # Translation button
        translate_button = st.button("🚀 Translate", type="primary", use_container_width=True)
        
        # Translation output area
        translation_placeholder = st.empty()
        
        if translate_button and urdu_input.strip():
            with st.spinner("🔄 Translating..."):
                try:
                    # Normalize input if requested
                    if normalize_input:
                        processed_input = normalize_urdu(
                            urdu_input,
                            remove_tashkeel=remove_tashkeel,
                            remove_tatweel=remove_tatweel,
                            normalize_alef=normalize_alef,
                            normalize_ye=normalize_ye,
                            normalize_kaf=normalize_kaf
                        )
                    else:
                        processed_input = urdu_input
                    
                    # Translate
                    if use_beam_search:
                        translation = beam_search_translate(
                            model, processed_input, sp_ur, sp_ro, 
                            beam_size=beam_size, max_len=max_length, device=device
                        )
                        method = f"Beam Search (size={beam_size})"
                    else:
                        translation = translate_sentence_greedy(
                            processed_input, model, sp_ur, sp_ro, 
                            max_len=max_length, device=device
                        )
                        method = "Greedy Search"
                    
                    # Display translation
                    translation_placeholder.text_area(
                        "Translation:",
                        value=translation,
                        height=250,
                        disabled=True
                    )
                    
                    # Add to history
                    st.session_state.translation_history.insert(0, {
                        'urdu': urdu_input,
                        'roman': translation,
                        'method': method,
                        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
                    })
                    
                    # Success message
                    st.success(f"✅ Translated using {method}")
                    
                    # Show processing details
                    with st.expander("🔍 Translation Details"):
                        col_det1, col_det2 = st.columns(2)
                        with col_det1:
                            st.write("**Original Text:**")
                            st.code(urdu_input, language=None)
                            if normalize_input:
                                st.write("**Normalized Text:**")
                                st.code(processed_input, language=None)
                        with col_det2:
                            st.write("**Translation Method:**", method)
                            st.write("**Max Length:**", max_length)
                            st.write("**Device:**", str(device).upper())
                            if normalize_input:
                                st.write("**Normalization:** Enabled")
                    
                except Exception as e:
                    st.error(f"❌ Translation failed: {str(e)}")
        
        elif translate_button and not urdu_input.strip():
            st.warning("⚠️ Please enter some Urdu text to translate.")
        else:
            translation_placeholder.text_area(
                "Translation will appear here...",
                value="",
                height=250,
                disabled=True
            )
    
    # Example section
    st.markdown("---")
    st.subheader("💡 Try These Examples")
    
    examples = [
        "خدا کی خدائی میں تجھ سا نہ دیکھا۔",
        "مجھے پاکستان سے پیار ہے۔", 
        "تمہاری مسکراہٹ دل کو بہار کا پیغام دیتی ہے۔",
        "شکریہ آپ کا بہت بہت۔",
        "کیا قیامت ہے آپ کا تماشا",
        "میں تم سے محبت کرتا ہوں۔"
    ]
    
    # Create example buttons in rows
    cols_per_row = 3
    for i in range(0, len(examples), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(examples):
                example_text = examples[i + j]
                with col:
                    if st.button(f"Example {i+j+1}", key=f"ex_{i+j}", use_container_width=True):
                        st.session_state.current_input = example_text
                        st.rerun()
                    # Show preview
                    st.caption(example_text[:30] + "..." if len(example_text) > 30 else example_text)
    
    # Translation history
    if st.session_state.translation_history:
        st.markdown("---")
        st.subheader("📚 Translation History")
        
        # Show recent translations
        history_to_show = st.selectbox("Show recent translations:", [5, 10, 20], index=0)
        
        for i, item in enumerate(st.session_state.translation_history[:history_to_show]):
            with st.expander(f"🕒 {item['timestamp']} - {item['method']}"):
                col_hist1, col_hist2 = st.columns(2)
                with col_hist1:
                    st.write("**Urdu:**")
                    st.write(item['urdu'])
                with col_hist2:
                    st.write("**Roman:**")
                    st.write(item['roman'])
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        🧠 <strong>Neural Machine Translation</strong><br>
        PyTorch Seq2Seq with Attention • SentencePiece Tokenization<br>
        Built with ❤️ using Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

# Import pandas for timestamp
import pandas as pd

if __name__ == "__main__":
    main()