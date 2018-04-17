import numpy as np


def detect_and_disply(model,img,return_mode="thres",verbose = 1):
    results = model.detect([img],return_mode=return_mode)

    r = results
    if r is None:
        print("No instance has been detect")
        return None,None
    else:
        r = r[0]
    if verbose == 1:
        print("total nuclei detected: ",r['masks'].shape[2])
        #visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
        #                    dataset_val.class_names, r['scores'], ax=get_ax())
    #print(r['masks'].dtype)
    return r['masks'],r['scores']


def find_match(l_ind,r_ind,l_mask,r_mask,threshold = 20):
    match = []
    for l in  l_ind:
        for r in r_ind:
            new = l_mask[:,:,l] + r_mask[:,:,r]
            if np.count_nonzero(new>1)>threshold:
                match.append((l,r))
                #r_ind.remove(r)
                break
    return match   

def expand_unmatch(a,a_score,ind,raw_shape,mode="lr"):
    a = a[:,:,ind]
    a_score = a_score[ind]
    assert mode in {"ud","du","lr","rl"}
    height = a.shape[0]
    raw_height = raw_shape[0]
    width = a.shape[1]
    raw_width =raw_shape[1]
    num = a.shape[2]
    if mode == "ud":
        new_mask = np.zeros((raw_height,raw_width,num))
        new_mask[:height,:,:] +=  a[:,:,:]
    elif mode == "du":
        new_mask = np.zeros((raw_height,raw_width,num))
        new_mask[raw_height-height:,:,:] +=  a[:,:,:]
    elif mode == "lr":
        new_mask = np.zeros((raw_height,raw_width,num))
        new_mask[:,:width,:] +=  a[:,:,:]
    elif mode == "rl":

        new_mask = np.zeros((raw_height,raw_width,num))
        new_mask[:,raw_width-width:,:] +=  a[:,:,:]
    return new_mask,a_score

def combine_mask(a,b,score_a,score_b,match,raw_shape,mode="lr",binary=True):
    re_masks = []
    re_scores = []
    a_width = a.shape[1]
    b_width = b.shape[1]
    
    raw_height = raw_shape[0]
    raw_width =raw_shape[1]
    
    for single in match:
    
        l = single[0]
        r = single[1]
        new_mask = np.zeros((raw_height,raw_width))
        new_mask[:,:a_width] +=  a[:,:,l]
        new_mask[:,raw_width-b_width:] += b[:,:,r]

        if binary:
            new_mask[np.where(new_mask == 2.0)] = 1.0
        else :
            new_mask[:,raw_width-b_width:a_width] = new_mask[:,raw_width-b_width:a_width]/2
            
        """
        plt.imshow(a[:,:,l])
        
        plt.show()
        plt.imshow(b[:,:,r])
        plt.show()
        plt.imshow(new_mask)
        plt.show()
        """

        
        #print(score_a.shape)
        #print(score_b.shape)
        re_scores.append((score_a[l]+score_b[r])/2)
        re_masks.append(new_mask)
    #print("re_scores: " ,re_scores)
    #print("np re_scores: ",np.array(re_scores))
    #print("re_scores: " ,len(re_scores),re_scores[0].shape)
    #print("re_masks: " ,len(re_masks),re_masks[0].shape)    
    re_scores = np.array(re_scores)
    re_masks = np.transpose(np.array(re_masks),[1,2,0])
    #print("re_scores: " ,re_scores.shape)
    #print("re_masks: " ,re_masks.shape)
    return re_masks , re_scores

def divide_recursive_detect(model,img,inter_width,max_edge,combine_threshold,return_mode,verbose = 0):
    miss_a_flag = 0
    miss_b_flag = 0
    a_match = []
    b_match = []
    match = []
    raw_shape = img.shape
    
    if raw_shape[0] <= max_edge and raw_shape[1] <= max_edge:
        return detect_and_disply(model,img,return_mode,verbose = verbose)
    
    elif raw_shape[1] > max_edge:

        cscore = np.zeros((1))
        expa_score = np.zeros((1))
        expb_score = np.zeros((1))
        
        cmask = np.zeros((raw_shape[0],raw_shape[1],1))
        expb = np.zeros((raw_shape[0],raw_shape[1],1))
        expa =  np.zeros((raw_shape[0],raw_shape[1],1))
    
        a_width = int(raw_shape[1]/2)
        b_width = raw_shape[1] - a_width +inter_width
        if verbose:
            print("the width is",raw_shape[1])
            print("divding in to ",a_width,"and ",b_width)
            

        left = img[:,:a_width,:]
        right = img[:,raw_shape[1]-b_width:,:]
        
        if verbose:
            plt.imshow(left)
            plt.show()
            plt.imshow(right)
            plt.show()
        a,sa = divide_recursive_detect(model,left,inter_width,max_edge,combine_threshold,return_mode,verbose)
        if a is not None:
            #print("1",type(a))
            inter_a = a[:,a_width-inter_width:a_width,:]
            mask_height = np.sum(np.any(inter_a, axis=0), axis=0)
            mask_width = np.sum(np.any(inter_a, axis=1), axis=0)
            flag = (mask_height > 2) * (mask_width > 2) 
            #print("2",flag)
    
            inter_ind_l =  list(np.argwhere(flag == True).squeeze(1))
            empty_ind_l =  list(np.argwhere(flag == False).squeeze(1))



        else :
            miss_a_flag +=1
        

        b,sb = divide_recursive_detect(model,right,inter_width,max_edge,combine_threshold,return_mode,verbose)
        if b is not None:

            inter_b  = b[:,:inter_width,:]
            mask_height = np.sum(np.any(inter_b, axis=0), axis=0)
            mask_width = np.sum(np.any(inter_b, axis=1), axis=0)
            flag = (mask_height > 2) * (mask_width > 2) 
    
            inter_ind_r =  list(np.argwhere(flag == True).squeeze(1))
            empty_ind_r =  list(np.argwhere(flag == False).squeeze(1))


        else:
            miss_b_flag +=1
            
            
        if miss_a_flag+miss_b_flag == 2:
            return None,None
        
        if  miss_a_flag+miss_b_flag == 0 :
            
            match = find_match(inter_ind_l,inter_ind_r,inter_a,inter_b,combine_threshold)
            #cmask= combine_mask(a,b,match,img.shape[0:2],mode="lr")
            #print("find ",len(match), " match!!!!")
            if len(match) != 0:
                #print("start combining")
                if return_mode == "thres":
                    binary = True
                else:
                    return_mode == "raw"
                    binary = False
                cmask,cscore= combine_mask(a,b,sa,sb,match,raw_shape,mode="lr",binary=binary)
                #print("combining mask shape",cmask.shape,cscore.shape)
                #print("combine mask",cmask.sum())
                a_match = list(zip(*match))[0]
                b_match = list(zip(*match))[1]


            
        if miss_a_flag == 0:
            
            a_unmatch = set(empty_ind_l)|set(inter_ind_l) - set(a_match)
            expa,expa_score = expand_unmatch(a,sa,list(a_unmatch),raw_shape[0:2],mode="lr")
            

        if miss_b_flag == 0 :
            
            b_unmatch = set(empty_ind_r)|set(inter_ind_r) - set(b_match)
            #print(b.dtype)
            expb,expb_score = expand_unmatch(b,sb,list(b_unmatch),raw_shape[0:2],mode="rl")
        """
        print(expa.dtype)
        print(expb.dtype)
        print(cmask.dtype)
        """
        if return_mode == "thres":
            return np.concatenate((expa,expb,cmask),axis=2).astype(np.uint8), np.concatenate((expa_score,expb_score,cscore),axis=0).astype(np.uint8)
        elif return_mode == "raw":
            return np.concatenate((expa,expb,cmask),axis=2).astype(np.float32), np.concatenate((expa_score,expb_score,cscore),axis=0).astype(np.float32)
    elif raw_shape[0] > max_edge:
        img = np.transpose(img,[1,0,2])
        ret_mask,ret_score = divide_recursive_detect(model,img,inter_width,max_edge,combine_threshold,return_mode,verbose)
        if ret_mask is not None:
            return np.transpose(ret_mask,[1,0,2]),ret_score
        else:
            return None,None




