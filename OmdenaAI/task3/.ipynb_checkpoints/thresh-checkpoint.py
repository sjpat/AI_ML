# Get thresholds for beginner squats

def get_thresholds_squat_beginner():
    '''
    Function to get squat thresholds for beginners
    '''
    _ANGLE_HIP_KNEE_VERT = {
        'NORMAL' : (0,32),
        'TRANS' : (35,65),
        'PASS' : (70,95)
    }

    thresholds = {
        'HIP_KNEE_VERT' : _ANGLE_HIP_KNEE_VERT,
        'HIP_THRESH' : [10, 50],
        'ANKLE_THRESH' : 45,
        'KNEE_THRESH' : [50,70,95],

        'OFFSET_THRESH' : 35.0,
        'INACTIVE_THRESH' : 15.0,

        'CNT_FRAME_THRESH' : 50
    }

return thresholds

def get_thresholds_squat_pro():
    '''
    Function to get squat thresholds for professionals
    '''
    
    _ANGLE_HIP_KNEE_VERT = {
        'NORMAL' : (0,32),
        'TRANS' : (35,65),
        'PASS' : (80,95)
    }
    
    thresholds = {
        'HIP_KNEE_VERT': _ANGLE_HIP_KNEE_VERT,
        'HIP_THRESH' : [15,50],
        'ANKLE_THRESH' : 30,
        'KNEE_THRESH' : [50,80,95],
        
        'OFFSET_THRESH' : 35.0,
        'INACTIVE_THRESH' : 15.0,
        
        'CNT_FRAME_THRESH' : 50
                            
                 }
                 
return thresholds

def get_thresholds_pushup_beginner():
    '''
    Function to get pushup thresholds for beginners
    '''
    _ANGLE_SHOULDER_ELBOW_VERT = {
        'NORMAL' : (),
        'TRANS' : (),
        'PASS' : ()
    }

    thresholds = {
        'SHOULDER_ELBOW_VERT' : _ANGLE_SHOULDER_ELBOW_VERT,
        'HIP_THRESH' : [],
        'ANKLE_THRESH' : 0.0,
        'KNEE_THRESH' : [],

        'OFFSET_THRESH' : 0.0,
        'INACTIVE_THRESH' : 0.0,

        'CNT_FRAME_THRESH' : 0
    }
    
def get_thresholds_pushup_pro():
    '''
    Function to get pushup thresholds for professionals
    '''
    _ANGLE_SHOULDER_ELBOW_VERT = {
        'NORMAL' : (),
        'TRANS' : (),
        'PASS' : ()
    }

    thresholds = {
        'SHOULDER_ELBOW_VERT' : _ANGLE_SHOULDER_ELBOW_VERT,
        'HIP_THRESH' : [],
        'ANKLE_THRESH' : 0.0,
        'KNEE_THRESH' : [],

        'OFFSET_THRESH' : 0.0,
        'INACTIVE_THRESH' : 0.0,

        'CNT_FRAME_THRESH' : 0
    }