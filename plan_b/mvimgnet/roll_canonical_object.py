"""Deterministic camera orientation before object framing; camera4D stays unchanged.

At the original camera center, rotate the virtual camera to look at the object
with the catalog's spherical meridian as image up. This exact ray homography
removes roll and aiming rotation, without assuming object depth or changing
azimuth/elevation. RGB/mask/valid share one source-to-output sampling map.
"""
import cv2
import numpy as np
from functools import lru_cache
from pathlib import Path
from PIL import Image
from .canonical_object import (direction_features, undistort_points,
    NonInvertibleDistortion, MissingCanonicalForeground, CONDITIONING_POLICY, SIZE, FILL)
from .colmap_io import qvec_to_rotmat, read_images_binary

PREPROCESS_POLICY='simple_radial_meridian_lookat_mask128_center_longedge80_gray128_v3'
ORIENTATION_POLICY='reference_image_up_spherical_meridian_raw_pose_v1'


def unit(vector):
    vector=np.asarray(vector,dtype=np.float64)
    norm=np.linalg.norm(vector)
    if not np.isfinite(norm) or norm<1e-8:
        raise ValueError('Degenerate orientation basis')
    return vector/norm


@lru_cache(maxsize=128)
def original_reference_pose(instance_path,image_id):
    # A reference RGB can be excluded by mask auditing; its pose still defines
    # every recorded direction. Never substitute another reference camera.
    rec=read_images_binary(Path(instance_path)/'sparse/0/images.bin')[image_id]
    return {'qvec':rec.qvec.tolist(),'tvec':rec.tvec.tolist()}


def reference_basis(row):
    ref=next((v for v in row['views'] if v['image_id']==row['reference_image_id']),None)
    if ref is None:ref=original_reference_pose(row['instance_path'],row['reference_image_id'])
    r=qvec_to_rotmat(np.asarray(ref['qvec'],dtype=np.float64))
    origin=np.asarray(row['object_center'],dtype=np.float64)
    x=unit(-r.T@np.asarray(ref['tvec'])-origin)
    up0=-r[1,:];up=up0-np.dot(up0,x)*x
    if np.linalg.norm(up)<1e-8:
        up=np.array([0.,0.,1.])-x[2]*x
    up=unit(up);y=unit(np.cross(up,x))
    return {'origin':origin,'x':x,'y':y,'up':up}


def orientation_rotation(view,basis):
    r=qvec_to_rotmat(np.asarray(view['qvec'],dtype=np.float64))
    d=unit(-r.T@np.asarray(view['tvec'])-basis['origin'])
    az,el=np.deg2rad([view['azimuth_deg'],view['elevation_deg']])
    horizontal=np.cos(az)*basis['x']+np.sin(az)*basis['y']
    expected=np.cos(el)*horizontal+np.sin(el)*basis['up']
    if np.linalg.norm(d-expected)>1e-5:
        raise ValueError('Catalog direction and raw pose disagree')
    # Analytic meridian tangent: finite even at +/-90 degree elevation.
    # At the exact pole the recorded azimuth selects the meridian explicitly.
    right=-np.sin(az)*basis['x']+np.cos(az)*basis['y']
    camera_up=-np.sin(el)*horizontal+np.cos(el)*basis['up']
    desired=np.stack([right,-camera_up,-expected])
    rotation=desired@r.T
    assert np.allclose(rotation@rotation.T,np.eye(3),atol=1e-6)
    assert np.linalg.det(rotation)>0.999999
    return rotation,desired


def rotate_pixels(points,params,rotation):
    f,cx,cy,_=params
    rays=np.column_stack([(points-np.array([cx,cy]))/f,np.ones(len(points))])
    transformed=rays@rotation.T
    if np.any(transformed[:,2]<=1e-5):
        raise MissingCanonicalForeground('Foreground crosses canonical camera horizon')
    return transformed[:,:2]/transformed[:,2,None]*f+[cx,cy]


def canonicalize(rgb,mask,params,rotation,*,size=SIZE,fill=FILL):
    rgb,mask=np.asarray(rgb),np.asarray(mask)
    if rgb.dtype!=np.uint8 or mask.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or mask.shape!=rgb.shape[:2]:
        raise ValueError('Expected matching uint8 RGB and grayscale mask')
    if size<16 or not 0<fill<1:raise ValueError('Invalid output framing')
    rotation=np.asarray(rotation,dtype=np.float64)
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all():raise ValueError('Invalid camera rotation')
    binary=np.uint8(mask>=128)
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    if not contours:raise MissingCanonicalForeground('Empty foreground')
    points=np.concatenate(contours).reshape(-1,2).astype(np.float64)
    edges=(points[:,None,:]+np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])).reshape(-1,2)
    effective=np.array(params,dtype=np.float64,copy=True)
    status='corrected' if effective[3]!=0 else 'zero_distortion'
    try:undistorted=undistort_points(edges,effective)
    except NonInvertibleDistortion:
        effective[3]=0;undistorted=undistort_points(edges,effective)
        status='identity_fallback_noninvertible_calibration'
    oriented=rotate_pixels(undistorted,effective,rotation)
    lo,hi=oriented.min(0),oriented.max(0);center=(lo+hi)/2
    side=float(max(hi-lo)/fill)
    if not np.isfinite(side) or side<=0:raise ValueError('Invalid oriented foreground extent')
    f,cx,cy,k=effective;n=size*2
    axis=(np.arange(n,dtype=np.float64)+.5)*side/n-side/2
    ux,uy=np.meshgrid(axis+center[0],axis+center[1])
    rays=np.stack([(ux-cx)/f,(uy-cy)/f,np.ones_like(ux)],axis=-1)@rotation
    forward=rays[...,2]>1e-5
    safe_z=np.where(forward,rays[...,2],1.)
    x,y=rays[...,0]/safe_z,rays[...,1]/safe_z
    r2=x*x+y*y;factor=1+k*r2
    map_x,map_y=(f*x*factor+cx).astype(np.float32),(f*y*factor+cy).astype(np.float32)
    invalid=~forward|((1+3*k*r2)<=0)|~np.isfinite(map_x)|~np.isfinite(map_y)
    map_x[invalid]=map_y[invalid]=-1
    source=np.where(binary[...,None]!=0,rgb,np.uint8(128))
    color=cv2.remap(source,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(128,128,128))
    fg=cv2.remap(binary.astype(np.float32),map_x,map_y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    valid=cv2.remap(np.ones_like(binary),map_x,map_y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    color=cv2.resize(color,(size,size),interpolation=cv2.INTER_AREA)
    fg=cv2.resize(fg,(size,size),interpolation=cv2.INTER_AREA)
    valid=cv2.resize(valid.astype(np.float32),(size,size),interpolation=cv2.INTER_AREA)
    if fg.max()==0:raise MissingCanonicalForeground('Foreground vanished during canonicalization')
    info={'policy':PREPROCESS_POLICY,'orientation_policy':ORIENTATION_POLICY,
        'camera_rotation_original_to_canonical':rotation.tolist(),'oriented_bbox':[*lo.tolist(),*hi.tolist()],
        'crop_center':center.tolist(),'crop_side':side,'output_size':size,'fill':fill,
        'source_size':[rgb.shape[1],rgb.shape[0]],'source_foreground_touches_edge':bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any()),
        'perspective_normalized':False,'camera_aim_normalized':True,'roll_normalized':True,
        'distortion_status':status,'source_distortion_k':float(params[3]),'effective_distortion_k':float(k)}
    return color,fg,valid,info


def load_canonical_view(view,basis):
    if view['camera_model']!='SIMPLE_RADIAL':raise ValueError('Unsupported camera model')
    rotation,_=orientation_rotation(view,basis)
    with Image.open(view['path']) as im,Image.open(view['mask_path']) as mask:
        if im.getexif().get(274,1)!=1 or mask.getexif().get(274,1)!=1:raise ValueError('Unexpected EXIF rotation')
        if mask.mode!='L' or mask.size!=im.size:raise ValueError('Expected matching grayscale mask')
        if im.size!=(view['resize']['source_width'],view['resize']['source_height']):raise ValueError('Camera source size mismatch')
        return canonicalize(np.array(im.convert('RGB')),np.array(mask),view['camera_params'],rotation)
