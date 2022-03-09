if [ "$1" = "nov" ]; then
    FIRST_DAY='20181112'
    LAST_DAY='20181131'
elif [ "$1" = "dec" ]; then
    FIRST_DAY='20181201'
    LAST_DAY='20181231'
elif [ "$1" = "jan" ]; then
    FIRST_DAY='20190101'
    #FIRST_DAY='20190125'
    LAST_DAY='20190131'
elif [ "$1" = "feb" ]; then
    FIRST_DAY='20190201'
    LAST_DAY='20190228'
elif [ "$1" = "all" ]; then
    FIRST_DAY='20181112'
    LAST_DAY='20190228'
else
    echo check input argument: $1
    exit
fi
DAY=$FIRST_DAY

N=4
until [[ ${DAY} > ${LAST_DAY} ]]; do
    ((i=i%N)); ((i++==0)) && wait
    #echo $DAY
#    python3 statistics_and_profiles.py -d $DAY -col 1 -colprofiles 1 -nmn 1 &
    python3 plot_obs.py -d $DAY --allSpectra 1 -nmn 0
    DAY=$(date -d "$DAY + 1 day" +%Y%m%d)
done

#non-parallel
until [[ ${DAY} > ${LAST_DAY} ]]; do
    #echo $DAY
    #python3 statistics_and_profiles.py -d $DAY -col 1 -colprofiles 1
    #python3 statistics_and_profiles.py -d $DAY
    python3 plot_obs.py -d $DAY --allSpectra 1 -nmn 0
    DAY=$(date -d "$DAY + 1 day" +%Y%m%d)
done
