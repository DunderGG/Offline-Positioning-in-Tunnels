set term gif animate delay 10 size 800, 640
set output "3dplot.gif"
do for [n=0:25] {
    splot [143420.18:143440.32][6394330.42:6394360.215][30:40] "logFile.txt" u 1:2:3 every ::::n w lp t sprintf("n=%i", n)
}
