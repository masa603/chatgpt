from collections import Counter
import numpy as np

def arrays(rows):
    return np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32),np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],np.float32)


def neighbour_weights(x,y):
    # Same five-neighbour label-consistency rule used in V7, without importing its training runner.
    distances=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    neighbours=np.argsort(distances,axis=1,kind='stable')[:,:5]
    result=[]
    for i,indices in enumerate(neighbours):
        unique,counts=np.unique(y[indices],axis=0,return_counts=True)
        choices=[(-int(count),min(float(distances[i,j]) for j in indices if np.array_equal(y[j],label)),tuple(label.tolist())) for label,count in zip(unique,counts)]
        mode=np.array(min(choices)[2],np.float32)
        error=float(np.sqrt(np.mean(((y[i]-mode)/255)**2)))
        result.append(max(.1,1/(1+(error/.15)**2)))
    return np.array(result,np.float32)


def row_weights(rows,robust=False):
    x,y=arrays(rows);q=neighbour_weights(x,y)
    count=Counter((r['group'],r['chart_id'],r['color_id']) for r in rows)
    groups={color:{r['group'] for r in rows if r['chart_id']=='train_3x3' and r['color_id']==color}
            for color in {r['color_id'] for r in rows if r['chart_id']=='train_3x3'}}
    values=[]
    for i,r in enumerate(rows):
        if r['chart_id']=='train_3x3':
            # Equal contribution per target color, then per calibration group.
            value=1/len(groups[r['color_id']])/count[r['group'],r['chart_id'],r['color_id']]
            if robust and r['group'] not in ['patch_session01_cal1','v7_capture01','v8_capture01']:value*=float(q[i])
            values.append(value)
        else:
            # Keep original sparse chart examples with positive, documented weight.
            values.append(.1*float(q[i]))
    return np.array(values,np.float32)
