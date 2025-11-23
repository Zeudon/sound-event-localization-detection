import torch
from model_crnn import SELD_CRNN
from model_conformer import SELD_Conformer
from model import SMRSELDWithCSPDarkNet
from config import Config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_models():
    config = Config()
    
    # Dimensions
    B = 2
    T = 250
    C = 4
    F = 64
    I, J = 18, 36
    grid_size = I * J
    M = 14
    
    print(f"Testing with Input: [{B}, {T}, {C}, {F}]")
    print(f"Expected Output:  [{B}, {T}, {grid_size}, {M}]")
    print("-" * 50)
    
    # Dummy input
    x = torch.randn(B, T, C, F)
    
    # --- Test CNN (Original) ---
    print("Testing CNN (Original)...")
    try:
        model_cnn = SMRSELDWithCSPDarkNet(
            n_channels=C,
            grid_size=(I, J),
            num_classes=M,
            use_small=False
        )
        # CNN expects (B, T, C, F) but reshapes internally
        y_cnn = model_cnn(x)
        print(f"CNN Output:       {list(y_cnn.shape)}")
        print(f"CNN Parameters:   {count_parameters(model_cnn):,}")
        
        expected_shape = [B, T, grid_size, M]
        if list(y_cnn.shape) == expected_shape:
            print("✅ CNN Dimensions Correct")
        else:
            print(f"❌ CNN Dimensions Incorrect! Expected {expected_shape}, got {list(y_cnn.shape)}")
            
    except Exception as e:
        print(f"❌ CNN Failed: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 50)
    
    # --- Test CRNN ---
    print("Testing CRNN...")
    try:
        model_crnn = SELD_CRNN(
            n_channels=C,
            n_mels=F,
            grid_size=(I, J),
            num_classes=M,
            cnn_channels=[64, 128, 256, 512],
            rnn_hidden=256,
            rnn_layers=2
        )
        y_crnn = model_crnn(x)
        print(f"CRNN Output:      {list(y_crnn.shape)}")
        print(f"CRNN Parameters:  {count_parameters(model_crnn):,}")
        
        expected_shape = [B, T, grid_size, M]
        if list(y_crnn.shape) == expected_shape:
            print("✅ CRNN Dimensions Correct")
        else:
            print(f"❌ CRNN Dimensions Incorrect! Expected {expected_shape}, got {list(y_crnn.shape)}")
            
    except Exception as e:
        print(f"❌ CRNN Failed: {e}")
        import traceback
        traceback.print_exc()

    print("-" * 50)

    # --- Test Conformer ---
    print("Testing Conformer...")
    try:
        model_conf = SELD_Conformer(
            n_channels=C,
            n_mels=F,
            grid_size=(I, J),
            num_classes=M,
            cnn_channels=[64, 128, 256, 512],
            conf_d_model=256,
            conf_n_heads=4,
            conf_n_layers=2
        )
        y_conf = model_conf(x)
        print(f"Conformer Output: {list(y_conf.shape)}")
        print(f"Conformer Params: {count_parameters(model_conf):,}")
        
        expected_shape = [B, T, grid_size, M]
        if list(y_conf.shape) == expected_shape:
            print("✅ Conformer Dimensions Correct")
        else:
            print(f"❌ Conformer Dimensions Incorrect! Expected {expected_shape}, got {list(y_conf.shape)}")
            
    except Exception as e:
        print(f"❌ Conformer Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_models()
