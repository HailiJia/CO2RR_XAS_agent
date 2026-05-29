#!/bin/bash

parallel=-1
normal=-1
all=-1
tauc=-1
trace=-1
while [[ $# -gt 0 ]]
do
   key="$1"
   case $key in
      -parallel) parallel=0
      ;;
      -normal) normal=0
      ;;
      -trace) trace=0
      ;;
      -tauc) tauc=0
      ;;
   esac
   shift
done


cat > helpscript.perl  <<EOF
#!/bin/perl

use strict;
use warnings;
my \$mode=shift;

while(<>)
{
   chomp;
   if(\$_ =~ /frequency dependent IMAGINARY DIELECTRIC FUNCTION/)
   {
      \$_=<>;
      \$_=<>;
      while (<>)
      {
         my \$sum=0;
         if (\$_ !~ /[0-9]/) {last;}
         chomp;
         \$_=~s/^/ /;
         my @help=split(/[\t,\s]+/);
         if (\$help[2]=~/NaN/||\$help[3]=~/NaN/||\$help[4]=~/NaN/) {next;}
         if (\$help[5]=~/NaN/||\$help[6]=~/NaN/||\$help[4]=~/NaN/) {next;}
         if (\$mode==0) {\$sum=\$help[2]+\$help[3]+\$help[4]+\$help[5]+\$help[6]+\$help[7];}
         if (\$mode==1) {\$sum=\$help[4];}
         if (\$mode==2) {\$sum=\$help[2]+\$help[3];}
         if (\$mode==3) {\$sum=\$help[2]+\$help[3]+\$help[4];}
         if (\$mode==4) {\$sum=(\$help[1]*\$help[1]*(\$help[2]+\$help[3]+\$help[4]+\$help[5]+\$help[6]+\$help[7]))**0.5;}
         if (\$mode==5) {\$sum=(\$help[1]*\$help[1]*(\$help[4]))**0.5;}
         if (\$mode==6) {\$sum=(\$help[1]*\$help[1]*(\$help[2]+\$help[3]))**0.5;}
         if (\$mode==7) {\$sum=(\$help[1]*\$help[1]*(\$help[2]+\$help[3]+\$help[4]))**0.5;}
         print \$help[1]," ",\$sum,"\n";
      }
   }
   last if eof;
}
EOF

if [[ $normal -eq 0 ]]; then
   if [[ $tauc -eq 0 ]]; then
      perl helpscript.perl 4 OUTCAR > CORE_DIELECTRIC_IMAG.dat
   else
      perl helpscript.perl 1 OUTCAR > CORE_DIELECTRIC_IMAG.dat
   fi
else
   if [[ $parallel -eq 0 ]]; then
      if [[ $tauc -eq 0 ]]; then
         perl helpscript.perl 5 OUTCAR > CORE_DIELECTRIC_IMAG.dat
      else
         perl helpscript.perl 2 OUTCAR > CORE_DIELECTRIC_IMAG.dat
      fi
   else
      if [[ $trace -eq 0 ]]; then
         if [[ $tauc -eq 0 ]]; then
            perl helpscript.perl 6 OUTCAR > CORE_DIELECTRIC_IMAG.dat
         else
            perl helpscript.perl 3 OUTCAR > CORE_DIELECTRIC_IMAG.dat
         fi
      else
         if [[ $tauc -eq 0 ]]; then
            perl helpscript.perl 7 OUTCAR > CORE_DIELECTRIC_IMAG.dat
         else
            perl helpscript.perl 0 OUTCAR > CORE_DIELECTRIC_IMAG.dat
         fi
      fi
   fi
fi
rm helpscript.perl

