import pytest
import torch
from utils.glip_compat import modulated_deform_conv, ml_nms


def test_class_nms_preserves_separate_categories():
    boxes=torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.],[0.,0.,10.,10.]])
    keep=ml_nms(boxes,torch.tensor([.9,.8,.7]),torch.tensor([0,0,1]),.5)
    assert keep.tolist()==[0,2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_deformable_convolution_zero_offset_matches_regular_convolution():
    torch.manual_seed(77)
    x=torch.randn(1,2,8,8,device='cuda')
    weight=torch.randn(3,2,3,3,device='cuda')
    output=modulated_deform_conv(x,torch.zeros(1,18,8,8,device='cuda'),
                                torch.ones(1,9,8,8,device='cuda'),weight,padding=1)
    assert torch.allclose(output,torch.nn.functional.conv2d(x,weight,padding=1),atol=1e-5)


def test_legacy_oversized_offsets_match_flat_kernel_addressing():
    # A 1x1 kernel isolates offset addressing from kernel/channel aggregation.
    import torch.nn.functional as F
    x=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
    offset=torch.linspace(0,.3,128).reshape(1,2,8,8)
    mask=torch.linspace(.5,1.,64).reshape(1,1,8,8)
    result=modulated_deform_conv(x,offset,mask,torch.ones(1,1,1,1))
    expected=torch.zeros_like(x)
    for y in range(4):
        for col in range(4):
            # Original C kernel: channel * height_out * width_out + y * width_out + x.
            dy=offset.flatten()[y*4+col]
            dx=offset.flatten()[16+y*4+col]
            grid=torch.tensor([[[[2*(col+dx)/3-1,2*(y+dy)/3-1]]]])
            value=F.grid_sample(x,grid,align_corners=True).item()
            expected[0,0,y,col]=value*mask.flatten()[y*4+col]
    assert torch.allclose(result,expected,atol=1e-5)
