import datetime

base_settings = \
   { "IFR"           : 0.005#Infection to fatality ratio. 0.4% is number from WHO
   , 'R0'            : 5.0#Baseline reproduction number. 6.2 is value from original paper Maier & Brockmann
   , "t_infectious"  : 8#Amount of days that one is infectious. 8 is value from original paper Maier & Brockmann
   , "p_immune"      : 1.0#If you were already infected in previous wave, what is probability of being immune now.
   , 'p_ICU'         : 0.15#Given Hospitalization, chance of ending up in ICU
   , 'time_ICU'      : 12#Average time a patient spends in ICU before release/death/regular hospi
   , 'time_H'        : 8.5#Average time a patient spends in the hospital (including the ones who end up in ICU)
   , "delay_ICU"     : 4#Average time a patients spends in regular hospital before transfer to ICU
   , 'delay_death'   : 14#Average time delay in days between hospitalization and death.
   , "pop"           : 11606426#Total population (Belgium) as of 2020-10-31
   , "start_date"    : datetime.date( 2020, 9, 10)#Start date for wave / epidemy
   , "end_date"      : datetime.date.today()-datetime.timedelta( days=0 )#End date to fetch data from (for wave selection)
   , "proj_date"     : datetime.date( 2020, 12, 20 )#Date to show projection of numbers
   }
