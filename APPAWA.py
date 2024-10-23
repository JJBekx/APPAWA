#=================================================================================================
#                    APPAWA: Analytical Partial Pixel Area Weight Assignment                      |
#                                                                                                 |
#                                                Written by Dr. John Jasper Bekx Sept. 2024 - ... |
# ------------------------------------------------------------------------------------------------|
#                                                                                                 |
# Worthy of note: All routines assume a pixel size of (1 x 1) a.u.²                               |
#                 Therefore, APPAWA calculates a fractional weight, not an absolute partial area. |
# ================================================================================================

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# Functions =======================================================================================
# Pixel-grid related ---
def get_pixelIndex( px_orig, N_px ): # (x, y) - NOT (row, col)
    # ---------------------------------------------------------
    # Provides the counting index of                           |
    # a given pixel defined by its origin (lower-left corner). |
    # Example for a 3x3 pixel grid:                            |
    #                         ___________                      |
    #                        |_6_|_7_|_8_|                     |
    #                        |_3_|_4_|_5_|                     |
    #                        |_0_|_1_|_2_|                     |
    #                        |                                 |
    #                        l-> origin at (0,0)               |
    # ---------------------------------------------------------
    x_px, y_px = px_orig
    return N_px * y_px + x_px

def get_pixelOrigin( px_index, N_px ): # (x, y) - NOT (row, col)
    # ----------------------------------------
    # Provides the inverse of get_pixelIndex. |
    # ----------------------------------------
    return ( int( px_index%N_px ), int( px_index/N_px ) )

# Circle related ---
def circ( x, R, x0, y0 ):
    #------------------------------------------
    # Provides the y-coordinates at x of       |
    # a circle at origin (x0,y0) and radius R. |
    # -----------------------------------------
    y_pos = (R**2. - ( x-x0 )**2.)**0.5 + y0
    y_neg = -(R**2. - ( x-x0 )**2.)**0.5 + y0
    return y_pos, y_neg

def intersec_x( R, x0, y0, px_y ): # px_y = integer denoting the row of the lower-left corner of the pixel
    #----------------------------------------------------
    # Provides the x-coordinates of the intersections of | 
    # the circle at origin (x0,y0) and radius R and      |
    # the pixel row defined by px_y.                     |
    # ---------------------------------------------------
    if( np.abs( y0-px_y ) <= R ):
        x1 = -(R**2. - ( px_y - y0 )**2.)**0.5 + x0
        x2 = (R**2. - ( px_y - y0 )**2.)**0.5 + x0
    else:
        x1 = np.nan
        x2 = np.nan
    return x1, x2
    
def intersec_y( R, x0, y0, px_x ): # px_x = integer denoting the col of the lower-left corner of the pixel
    #----------------------------------------------------
    # Provides the y-coordinates of the intersections of | 
    # the circle at origin (x0,y0) and radius R and      |
    # the pixel col defined by px_x.                     |
    # ---------------------------------------------------
    if( np.abs( x0-px_x ) <= R ):
        y1 = -(R**2. - ( px_x - x0 )**2.)**0.5 + y0
        y2 = (R**2. - ( px_x - x0 )**2.)**0.5 + y0
    else:
        y1 = np.nan
        y2 = np.nan
    return y1, y2
    
# Enclosed-area related ---
def Area_enclosedPixelMC( px_orig, R, x0, y0, N_MC=10000 ): # px_orig = (x, y) 
    # ------------------------------------------------------------------------
    # Numerical Monte Carlo integration of the area enclosed by               |
    # a circle at origin (x0, y0) and radius R and                            |
    # a pixel at origin px_orig denoting the lower-left corner of the pixel.  |
    # All lengths are normalized to a pixel length value.                     |
    # Therefore, the result is the fraction of the pixel area being enclosed. |
    # ------------------------------------------------------------------------
    px_x, px_y = px_orig
    x_rand = np.random.uniform(px_x, px_x + 1, N_MC)
    y_rand = np.random.uniform(px_y, px_y + 1, N_MC)
    sum_enclosedByCurve = 0
    for i in range(len(x_rand)):
        if( y_rand[i] >= y0 ):
            enclosed_ByCurve = ( y_rand[i] <= circ(x_rand[i], R, x0, y0)[0] ) # [0] = circle y values >= y0
        else:
            enclosed_ByCurve = ( y_rand[i] > circ(x_rand[i], R, x0, y0)[1] )  # [1] = circle y values < y0
        sum_enclosedByCurve += enclosed_ByCurve
    Area_px = 1.**2. # Assuming normalized pixel length throughout
    Area_MC = sum_enclosedByCurve * Area_px / N_MC
    return Area_MC

def Integral_curvedSegment( x, R, x0, y0, px_line, yes_flipXY=False ):
    # -------------------------------------------------------------------------------
    # Note: for this explanation, pretend px_line = px_y = px_origin row (*)         |
    # Analytical solution of the integral                                            |   
    # int dx [ sqrt( R² - (x-x0)² ) + (y0 - px_line) ]                               |  
    # ( + constant omitted )                                                         |
    # Area enclosed in pixel at (px_x, px_y) between x=x1 and x=x2 is I(x2) - I(x1), |
    # where I(x) = Integral_curvedSegment( x, R, x0, y0, px_line )                   |
    #                                                                                |
    # (*) yes_flipXY=True calculates the area for a pixel that is difficult to       |
    # handle as is, but which is easier if one does x <-> y.                         |
    # In that case px_line = px_x = px_origin col, but the integral stays the same.  |
    # -------------------------------------------------------------------------------
    # x0, y0 = x0 * int(1-int(yes_flipXY)) + y0 * int(yes_flipXY), x0 * int(yes_flipXY) + y0 * int(1-int(yes_flipXY))
    if( yes_flipXY ):
        x0, y0 = y0, x0
    yes_bottomHalf = (px_line < int(y0)) 
    term1 = ( (x-x0) * ( R**2. - (x-x0)**2. )**0.5 + R**2. * np.asin( (x-x0) / R ) ) / 2.
    term2 = (-1.)**int(yes_bottomHalf) * x * ( y0 - px_line - int(yes_bottomHalf) )
    return term1 + term2

def get_areaSegment( px_orig, intersec1, intersec2, R, x0, y0 ):
    px_x, px_y = px_orig
    x1, y1 = intersec1
    x2, y2 = intersec2

    yes_intersecTwoRows = ( type(y1)==int and type(y2)==int and max(y1,y2)-min(y1,y2)==1 )
    yes_intersecDoubleCol = ( type(x1)==int and type(x2)==int and x1==x2 and np.abs(y2-y1)<1 )
    yes_y0ThroughPixel = ( min(y1,y2) < y0 and max(y1,y2) > y0)
    yes_integrateAfterFlip = yes_intersecTwoRows or yes_intersecDoubleCol or yes_y0ThroughPixel

    yes_integrateNormally = not yes_integrateAfterFlip

    if( yes_integrateNormally ): 
        insec1, insec2 = sorted( [ intersec1, intersec2 ], key=lambda elem:elem[0] ) # sort along x-coordinate
        x1, y1 = insec1
        x2, y2 = insec2
        px_line = px_y
        yes_flip = False
    elif( yes_integrateAfterFlip ): # flip x and y to just use the same integral 
        insec1, insec2 = sorted( [ intersec1, intersec2 ], key=lambda elem:elem[1] ) # sort along y-coordinate
        y1, x1 = insec1 # flip x & y
        y2, x2 = insec2
        px_line = px_x
        yes_flip = True # Needed in the integral to flip x0 and y0
    else: 
        print( "ERROR in get_areaSegment | Impossible case encountered for pixel at origin " + px_orig )
    area = Integral_curvedSegment( x2, R, x0, y0, px_line, yes_flip ) - Integral_curvedSegment( x1, R, x0, y0, px_line, yes_flip )

    # If two opposite corners are confined to the circle, we are missing a little rectangular block, added here.
    yes_twoCornersInCirc1 = ((px_x-x0)**2. + (px_y-y0)**2. < R**2.) and ((px_x+1-x0)**2. + (px_y+1-y0)**2. < R**2.)
    yes_twoCornersInCirc2 = ((px_x+1-x0)**2. + (px_y-y0)**2. < R**2.) and ((px_x-x0)**2. + (px_y+1-y0)**2. < R**2.)
    yes_missingBlock = yes_twoCornersInCirc1 or yes_twoCornersInCirc2
    block = 0
    if( yes_missingBlock ):
        yes_rightOfx0 = ( type(x1)!=int )
        xc = x1 * int( yes_rightOfx0 ) + x2 * int( 1-yes_rightOfx0 )
        block = (-1.)**int( 1-yes_rightOfx0 ) * (xc - px_x) + int( 1-yes_rightOfx0 )
    area += block

    return area

# Checking test cases ---
def TestCase(c): #returns (R, x0, y0)
    if( c=="A" ):
        R = 2.
        x0 = 5.
        y0 = 5.
    elif( c=="B" ):
        R = 2.
        x0 = 5.5
        y0 = 5.
    elif( c=="C" ):
        R = 2.
        x0 = 5.
        y0 = 5.5
    elif( c=="D" ):
        R = 2.
        x0 = 5.5
        y0 = 5.5
    elif( c=="E" ):
        R = 2.5
        x0 = 5.
        y0 = 5.
    elif( c=="F" ):
        R = 2.5
        x0 = 5.5
        y0 = 5.
    elif( c=="G" ):
        R = 2.5
        x0 = 5.
        y0 = 5.5
    elif( c=="H" ):
        R = 2.5
        x0 = 5.5
        y0 = 5.5
    elif( c=="I" ):
        R = math.pi
        x0 = 5.1111
        y0 = 5.1415
    return (R, x0, y0)

# Azimuthal line related ---
def Line( x, chi, x0, y0 ):
    if( math.isclose(np.abs(chi), math.pi) ): 
        return np.nan
    else:
        return y0 - ( x - x0 ) / math.tan( chi-math.pi/2. )

def intersec_LineX( chi, x0, y0, px_y ): # px_y = integer denoting the row of the lower-left corner of the pixel
    if( math.isclose(np.abs(chi), math.pi) ): # If the Line is vertical, it intersects the row px_y at px_y
        return px_y
    else:
        return x0 - ( px_y - y0 ) * math.tan( chi-math.pi/2. ) 

def intersec_LineY( chi, x0, y0, px_x ): # px_x = integer denoting the col of the lower-left corner of the pixel
    if( math.isclose(np.abs(chi), math.pi) ): # If the Line is vertical, it intersects the col px_x either never or inf many times
        return np.nan
    else:
        return y0 - ( px_x - x0 ) / math.tan( chi-math.pi/2. )

# =================================================================================================

# Parameters ======================================================================================
# Pixel grid ---
N_px     = 10 # = (N_px x N_px) grid 
px_len   = 1. # Pixel size (width=height) - mostly ignored
grid_min = 0
grid_max = N_px # Normalized to 1 x 1 (a.u.²) pixels. 
grid_sq  = range( grid_min, grid_max+1 ) # endpoint inclusive = N_px + 1 points
# grid_sq = np.linspace( grid_min, grid_max, num=N_px+1, endpoint=True, dtype=int ) # linspace uses np.int64 (=slow) instead of native int (=fast)

# Circle ---
# R  = math.pi / px_len # Circle radius in pixel units
# x0 = 5.1111 / px_len  # N_px * px_len / 2. # Circle origin 
# y0 = 10-5.1415 / px_len  # N_px * px_len / 2. 

R, x0, y0 = TestCase("E")

# Azimuthally defined lines --- 
chi = math.pi / 4.12 # Defined according to mathematicians' conventions.

# Main ============================================================================================
t0 = time.time()

# Identify ROI with circle in it ---
ROI_ixMin = int( (x0 - R) )
ROI_ixMax = int( (x0 + R) + 0.99 ) # TODO: Could be prone to bugs - Is a tight ROI square absolutely necessary? 
ROI_x     = range( ROI_ixMin, ROI_ixMax+1 ) # ROI_x = np.linspace( ROI_ixMin, ROI_ixMax, num=ROI_ixMax-ROI_ixMin+1, endpoint=True,dtype=int ) 
ROI_jyMin = int( (y0 - R) )
ROI_jyMax = int( (y0 + R) + 0.99 ) 
ROI_y     = range( ROI_jyMin, ROI_jyMax+1 )# ROI_y = np.linspace( ROI_jyMin, ROI_jyMax, num=ROI_jyMax-ROI_jyMin+1, endpoint=True, dtype=int )

# All pixel-circle intersections within the ROI & identifying to which pixels they correspond to ---
intersec        = [] # All intersection locations as tuples.
px_origIntersec = [] # Pixel origins of pixels that contain the intersections.
dict_pxIntersec = {} # Dictionary = { "pixel index with an intersection": [ intersection1, intersection2 ] }.

for i in ROI_x: # Gathering all intersections with x=integers
    y_D, y_U = intersec_y( R, x0, y0, i )
    if( math.isclose( y_U,y_D ) ): continue # You hit a crossing where y_D=y_U. Ignore this, it will be caught by x_L and x_R
    for y in y_D, y_U: # Classify the intersections
        if( not np.isnan(y) ): # checking y != np.nan leads to some spooky business - there's a difference between numpy nan and "true" nan apparently
            intersec.append( (i, y) ) 
            if( math.isclose( y, int(y) ) and not( math.isclose( y_D, ROI_y[0] ) or math.isclose( y_U, ROI_y[-1] ) ) ): # You hit a full-on four-way crossing. Needs some special treatment.
                yes_TLorBR = ( (i-x0)*(y-y0) < 0 ) # Either top left or bottom right quadrant
                for offset in 0,1:
                    px_orig = ( i + ( int(-offset)*int(yes_TLorBR) + int(offset-1)*int(1-yes_TLorBR) ), int(y) + int(-offset) )
                    px_origIntersec.append( px_orig )
                    px_index = get_pixelIndex( px_orig, N_px )
                    key = str(px_index)
                    dict_pxIntersec.setdefault(key, []).append( (i,y) )
            else:
                yes_needsOriginFix = (y==y_U) and ( math.isclose( y_U, ROI_y[-1]) ) # math.isclose is a bit more consistent than numpy.isclose
                for col in 1,0: # Just doing the pixels left (i-1) and right (i) of the y intersection in one loop
                    px_orig = (i-col,int(y) - int(yes_needsOriginFix))
                    px_origIntersec.append( px_orig )
                    px_index = get_pixelIndex( px_orig, N_px )
                    key = str(px_index)
                    dict_pxIntersec.setdefault(key, []).append( (i,y) )
    
for j in ROI_y:
    x_L, x_R = intersec_x( R, x0, y0, j )
    if( math.isclose( x_R,x_L ) ): continue # You hit a crossing where x_L=x_R. Ignore this, it was caught by y_D and y_U
    for x in x_L, x_R:
        if( not np.isnan(x) ):
            if( math.isclose( x, int(x) ) and not( math.isclose( x_L, ROI_x[0] ) or math.isclose( x_R, ROI_x[-1] ) ) ): continue # caught these in the y-loop above 
            intersec.append( (x, j) )
            yes_needsOriginFix = (x==x_R) and ( math.isclose( x_R, ROI_x[-1] ) )
            for row in 1,0: # Just doing the pixels below (j-1) and above (j) of the x intersection in one loop
                px_origIntersec.append( (int(x) - int(yes_needsOriginFix),j-row) )
                px_index = get_pixelIndex( (int(x) - int(yes_needsOriginFix),j-row), N_px )
                key = str(px_index)
                dict_pxIntersec.setdefault(key, []).append( (x,j) )

# Identifying pixels in full interior of circle ---
px_origIntersec = list( set(px_origIntersec) ) # Removing doubles
px_origIntersec = sorted( px_origIntersec )
px_origIntersecOrganized = [[px_origIntersec[0]]] # End result is [[(i, j), (i, j+1), ...], [(i+1, k), (i+1, k+1), ...], ...]
list_index = 0
for i in range(1,len(px_origIntersec)):
    curr_tup = px_origIntersec[i]
    prev_tup = px_origIntersec[i-1]
    if( curr_tup[0] == prev_tup[0] ):
        px_origIntersecOrganized[list_index].append(curr_tup)    
    else:
        list_index += 1
        px_origIntersecOrganized.append([curr_tup])

dict_pxAreaInside = {} # Pixels fully inside of the circle 
for lst in px_origIntersecOrganized:
    col  = lst[0][0]
    rows = [ tup[1] for tup in lst ]
    next_colIfPos = -1
    for step in range( min(rows), max(rows) + 1 ):
        if( step in rows ):
            if( next_colIfPos >=0 ):
                next_colIfPos = -1
                continue
            else:
                pass
        else: 
            next_colIfPos += 1
            px_index = get_pixelIndex( ( col, step ), N_px )
            key = str(px_index)
            dict_pxAreaInside[key] = 1. # Area is 1 a.u.²

# Calculating area weights ---
dict_pxAreaFrac = {}
for key, val in dict_pxIntersec.items():
    px_index = int(key)
    px_orig = get_pixelOrigin( px_index, N_px )
    assert len(val)<=4, "ERROR in classifying intersections | Found too many intersections in pixel " + key
    if( len(val)==2 ):  # In most cases, there are two intersections.
        area = get_areaSegment( px_orig, val[0], val[1], R, x0, y0 )
    elif( len(val)==3 ): # In a rare case, you may be dead-on grazing the pixel boundary at a single point. Just ignore this point.
        x_insec = [ i[0] for i in val ]
        y_insec = [ i[1] for i in val ]
        for i in range(len(val)):
            index1 = int(i/2) # Just looping over (0,1), (0,2), (1,2)
            index2 = int((i+1)/2)+1
            if( type(x_insec[index1]) == type(x_insec[index2]) ): 
                area = get_areaSegment( px_orig, val[index1], val[index2], R, x0, y0 )
            else:
                pass
    elif( len(val)==4 ): # Experience has found that four intersections is possible. 
        x_insec = [ i[0] for i in val ]
        y_insec = [ i[1] for i in val ]
        yes_sameIntersecX = ( len(x_insec)!=len(set(x_insec)) ) # Only one of these two
        yes_sameIntersecY = ( len(y_insec)!=len(set(y_insec)) ) # should be true
        assert yes_sameIntersecX != yes_sameIntersecY, "ERROR in classifying intersections | Impossible case encountered for pixel " + key
        sort_along = int(yes_sameIntersecX) 
        val_ordered = sorted( val, key=lambda elem:elem[ sort_along ] ) # sort along x/y if there are two intersections on the same pixel row/column
        area = 0
        area += get_areaSegment( px_orig, val_ordered[0], val_ordered[1], R, x0, y0 )
        area += get_areaSegment( px_orig, val_ordered[2], val_ordered[3], R, x0, y0 )
        area += int(yes_sameIntersecY) * np.abs( val_ordered[2][0] - val_ordered[1][0] ) + int(yes_sameIntersecX) * np.abs( val_ordered[2][1] - val_ordered[1][1] ) 
    else:
        print( "ERROR in classifying intersections - Expecting 2 or 4 | Pixel " + key + " has " + str(len(val)) + " intersections." )
        exit()
    assert(area <= 1.), "ERROR in calculating fractional area | Area is larger than 1 for pixel " + key
    dict_pxAreaFrac[key] = area

for k,v in dict_pxAreaFrac.items():
    px_orig=get_pixelOrigin(int(k), N_px)
    # MC = Area_enclosedPixelMC( px_orig, R, x0, y0, 2000000 )
    # rel = (1-np.abs( MC-v )/ v)*100
    print( k, px_orig, v)#, MC, rel )

t1 = time.time()
t_algo = t1 -t0

# Check ---
t0 = time.time()
Ans = np.pi * R**2.
t1 = time.time()
t_anal = t1 -t0

t0 = time.time()
total_sum = 0
total_MC  = 0
N_MC = 1000000
for d in dict_pxAreaInside, dict_pxAreaFrac:
    for k,v in d.items():
        total_sum += v
        # px_orig = get_pixelOrigin( int(k), N_px )
        # total_MC += Area_enclosedPixelMC( px_orig, R, x0, y0, N_MC )

t1 = time.time()
t_MC = t1 -t0

print( "The analytical closed-form answer is: ", Ans, " and took ", t_anal, " sec to calculate" )
print( "The analytical algorithmic answer is: ", total_sum, " and took ", t_algo, " sec to calculate" )
# print( "The numerical  Monte-Carlo answer is: ", total_MC, " and took ", t_MC, " sec to calculate" )
# =================================================================================================

# Plotting ========================================================================================
fig, ax = plt.subplots()

N_plt = 1000 + 1
x     = np.linspace(grid_min, grid_max, num=N_plt, endpoint=True)

# Pixel grid ---
[ ax.vlines( x=i, ymin=grid_min, ymax=grid_max, color="gray", ls="--" ) for i in grid_sq ]
[ ax.hlines( y=i, xmin=grid_min, xmax=grid_max, color="gray", ls="--" ) for i in grid_sq ]

# ROI highlight ---
[ ax.vlines( x=ROI_x[i], ymin=ROI_y[0], ymax=ROI_y[-1], color="k" ) for i in range(len(ROI_x)) ]
[ ax.hlines( y=ROI_y[i], xmin=ROI_x[0], xmax=ROI_x[-1], color="k" ) for i in range(len(ROI_y)) ]

# Circle ---
y_pos, y_neg = circ( x, R, x0, y0 )
ax.plot(x, y_pos, color="tab:blue")
ax.plot(x, y_neg, color="tab:blue")

# Intersections between grid and circle ---
for i in range(len(intersec)):
    ax.plot( intersec[i][0], intersec[i][1], "x", color="tab:orange" )

# Line ---
y_L = Line( x, chi, x0, y0 )
ax.plot( x, y_L, color="tab:green" )
ax.plot( x0, y0, "x", color="tab:green" )

# Intersections between grid and line ---
for i in grid_sq:
    ax.plot( i, intersec_LineY( chi, x0, y0, i ), "^", color="tab:orange" )
    ax.plot( intersec_LineX( chi, x0, y0, i ), i, "^", color="tab:orange" )

# Example pixel highlight ---
px_origX = 5 
px_origY = 3  
ax.plot( px_origX, px_origY, "bo" )
ax.vlines( x=px_origX, ymin=px_origY, ymax=px_origY + 1, color="red" ) 
ax.vlines( x=px_origX + 1, ymin=px_origY, ymax=px_origY + 1, color="red" ) 
ax.hlines( y=px_origY, xmin=px_origX, xmax=px_origX + 1, color="red" ) 
ax.hlines( y=px_origY + 1, xmin=px_origX, xmax=px_origX + 1, color="red" ) 

# Pixel index numbering ---
fmt_len = 1 + int( math.log10(N_px*N_px) )
fmt_str = "{:0" + str(fmt_len) + "d}"


plt.show()
