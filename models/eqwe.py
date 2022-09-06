import torch
gt_bboxes_per_image = torch.tensor([[337.7500, 323.5000,   6.6172,   5.9727],
                                   [584.5000, 285.7500,   8.6875,   6.7930],
                                   [584.5000, 285.7500,   6.7930,   8.6875]],)
                                
num_gt = gt_bboxes_per_image.size(0)
flag = (gt_bboxes_per_image[:,2]>=gt_bboxes_per_image[:,3])#.type(int)
cxcywh1 = gt_bboxes_per_image[flag,:]#w>=h
cxcywh2 = gt_bboxes_per_image[~flag,:]

f1 = torch.zeros(num_gt,2)
f2 = torch.zeros(num_gt,2)

a1 = cxcywh1[:,2]/2
b1 = cxcywh1[:,3]/2
c1 = (a1**2+b1**2)**(0.5)
f1[flag,:] = torch.cat([cxcywh1[:,0][:,None]-c1.unsqueeze(-1),cxcywh1[:,1][:,None]],dim=1)
f2[flag,:]= torch.cat([cxcywh1[:,0][:,None]+c1.unsqueeze(-1),cxcywh1[:,1][:,None]],dim=1)
    
    
a2 = cxcywh2[:,3]/2
b2 = cxcywh2[:,2]/2 
c2 = (a2**2+b2**2)**(0.5)
print(cxcywh2[:,0][:,None].size())
print((cxcywh2[:,1][:,None]-c2.unsqueeze(-1)).size())
print(c2)
f1[~flag,:] = torch.cat([cxcywh2[:,0][:,None],cxcywh2[:,1][:,None]-c2.unsqueeze(-1)],dim=1)
f2[~flag,:] = torch.cat([cxcywh2[:,0][:,None],cxcywh2[:,1][:,None]+c2.unsqueeze(-1)],dim=1)
print(f1)
print(f2)
Dis = []
for i in range (num_gt):
    dis = (torch.sum((torch.vstack([x_shifts_per_image,y_shifts_per_image],1) -f1[i])**2,dim=1))**(0.5)+\
        + (torch.sum((torch.vstack([x_shifts_per_image,y_shifts_per_image],1)-f2[i])**2,dim =1))**(0.5)
    Dis.append(dis<max(gt_bboxes_per_image[:,3][i],gt_bboxes_per_image[:,2][i]))######budui 
is_in_boxes = torch.vstack(Dis)

is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0#[n_gt,all_anchors]