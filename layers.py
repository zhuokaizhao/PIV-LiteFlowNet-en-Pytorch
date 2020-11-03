import torch

backwarp_tenGrid = {}

def backwarp(feat_tensor, flow_tensor):
	if str(flow_tensor.size()) not in backwarp_tenGrid:
		horizontal_tensor = torch.linspace(-1.0, 1.0, flow_tensor.shape[3]).view(1, 1, 1, flow_tensor.shape[3]).expand(flow_tensor.shape[0], -1, flow_tensor.shape[2], -1)
		vertical_tensor = torch.linspace(-1.0, 1.0, flow_tensor.shape[2]).view(1, 1, flow_tensor.shape[2], 1).expand(flow_tensor.shape[0], -1, -1, flow_tensor.shape[3])

		backwarp_tenGrid[str(flow_tensor.size())] = torch.cat([ horizontal_tensor, vertical_tensor ], 1).cuda()
	# end

	flow_tensor = torch.cat([ flow_tensor[:, 0:1, :, :] / ((feat_tensor.shape[3] - 1.0) / 2.0), flow_tensor[:, 1:2, :, :] / ((feat_tensor.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=feat_tensor, grid=(backwarp_tenGrid[str(flow_tensor.size())] + flow_tensor).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)