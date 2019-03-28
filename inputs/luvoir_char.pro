;###############################################################################
;+
; Code to select stars with candidate exoEarths for characterization 
; from 2-year survey with LUVOIR
;-
;###############################################################################
;
; LUVOIR colors
;
;###############################################################################

red = [0, 255, 246, 255, 186, 101, 6, 33, 29]
green=[0, 255, 153, 153*1.4, 216, 162, 72, 39, 27]
blue =[0, 255, 9, 9*1.4, 255, 255, 255, 133, 76]

tvlct, red, green, blue

;###############################################################################

readcol,'LUVOIR-Architecture_A-NOMINAL_OCCRATES-observations_AR.csv',$
	hip, dist, type, visit, del_t, img_time, stat_spec_time, spec_time,$
	eec_yield, junk, junk, junk, junk, junk, junk, junk, junk, junk, $ 
	junk, junk, junk, junk, junk, junk, $ 
	format='A,F,A,I,F,F,D,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F', $ 
	delimiter=',',skipline=1,/nan,/preserve_null

;###############################################################################
;
; Spectroscopy observations during 2-year LUVOIR-A exoEarth survey.
;
; This is Chris's recommended way of calculating the total time and distribution of 
; individual exposure times.
; It gives very stochastic results for the number of observations and their 
; distribution of exposure times. Can't be helped, I think.
;
;###############################################################################

obs = n_elements(hip)

n = 1
tot_count_arr = fltarr(n)
tot_deep_time_arr = fltarr(n)

FOR j=0,n-1 DO BEGIN

deep_time_arr = fltarr(1)
star_arr = strarr(1)
dist_arr = fltarr(1)
type_arr = strarr(1)

count = 0

FOR i=0,obs-1 DO BEGIN
  rand = randomu(seed, 1)
  IF (eec_yield[i] ge rand) THEN BEGIN
    deep_time_arr[count] = spec_time[i]
    star_arr[count] = hip[i]
    dist_arr[count] = dist[i]
    type_arr[count] = type[i]
    count = count + 1
    deep_time_arr = [deep_time_arr, [0.0]] 
    star_arr = [star_arr, [' ']]
    dist_arr = [dist_arr, [0.0]]
    type_arr = [type_arr, [' ']]
  ENDIF
ENDFOR

a = where(deep_time_arr ne 0.0)
tot_deep_time_arr[j] = total(deep_time_arr[a])
tot_count_arr[j] = count

ENDFOR 

print,'Total spec. time (days):', deep_time_arr[a]
print,'Min/max sample spec. times (days):',minmax(deep_time_arr[a])
print,'Mean num. spec. observations:',mean(tot_count_arr)
print,'Mean total spec. time (days):',mean(tot_deep_time_arr)

plot,histogram(deep_time_arr[a]),xtitle='Exposure Time (days)',ytitle='Number of Observations'

writecol_akir,'luvoir-A_stars.txt',star_arr[a]+',',dist_arr[a],','+type_arr[a], $
	head_txt='% HIP, distance, type for stars with candidate exoEarths from 2-year LUVOIR-A survey'
	
;###############################################################################
;
; Spectroscopy observations during 2-year LUVOIR-B exoEarth survey.
;
; Chris's recommendation is to take the brightest 3/5 of the LUVOIR-A target 
; list and draw from that. 
;
;###############################################################################

obs = n_elements(hip)

tmp_bright = sort(spec_time)
bright = tmp_bright[0:fix(obs*(3./5.))]
print,'# of observations of brightest targets:',n_elements(bright)

new_spec_time = spec_time[bright]
new_hip = hip[bright]
new_dist = dist[bright]
new_type = type[bright]

n = 1
tot_count_arr = fltarr(n)
tot_deep_time_arr = fltarr(n)

FOR j=0,n-1 DO BEGIN

deep_time_arr = fltarr(1)
star_arr = strarr(1)
dist_arr = fltarr(1)
type_arr = strarr(1)

count = 0

FOR i=0,n_elements(bright)-1 DO BEGIN
  rand = randomu(seed, 1)
  IF (eec_yield[i] ge rand) THEN BEGIN
    deep_time_arr[count] = new_spec_time[i]
    star_arr[count] = new_hip[i]
    dist_arr[count] = new_dist[i]
    type_arr[count] = new_type[i]
    count = count + 1
    deep_time_arr = [deep_time_arr, [0.0]] 
    star_arr = [star_arr, [' ']]
    dist_arr = [dist_arr, [0.0]]
    type_arr = [type_arr, [' ']]
  ENDIF
ENDFOR

a = where(deep_time_arr ne 0.0)
tot_deep_time_arr[j] = total(deep_time_arr[a])
tot_count_arr[j] = count

ENDFOR 

print,'Total spec. time (days):', deep_time_arr[a]
print,'Min/max sample spec. times (days):',minmax(deep_time_arr[a])
print,'Mean num. spec. observations:',mean(tot_count_arr)
print,'Mean total spec. time (days):',mean(tot_deep_time_arr)

plot,histogram(deep_time_arr[a]),xtitle='Exposure Time (days)',ytitle='Number of Observations'

writecol_akir,'luvoir-B_stars.txt',star_arr[a]+',',dist_arr[a],','+type_arr[a], $
	head_txt='% HIP, distance, type for stars with candidate exoEarths from 2-year LUVOIR-B survey'

END