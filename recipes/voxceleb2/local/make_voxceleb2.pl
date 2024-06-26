#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#
# Usage: make_voxceleb2.pl /export/voxceleb2 dev data/dev
#
# Note: This script requires ffmpeg to be installed and its location included in $PATH.

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-voxceleb2> <dataset> <path-to-data-dir>\n";
  print STDERR "e.g. $0 /export/voxceleb2 dev data/dev\n";
  exit(1);
}

($data_base, $dataset, $out_dir) = @ARGV;

if ("$dataset" ne "dev" && "$dataset" ne "test") {
  die "dataset parameter must be 'dev' or 'test'!";
}

opendir my $dh, "$data_base/$dataset/wav" or die "Cannot open directory: $!";
my @spkr_dirs = grep {-d "$data_base/$dataset/wav/$_" && ! /^\.{1,2}$/} readdir($dh);
closedir $dh;

if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir";
}

open(SPKR, ">", "$out_dir/utt2spk") or die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">", "$out_dir/wav.scp") or die "Could not open the output file $out_dir/wav.scp";

foreach (@spkr_dirs) {
  my $spkr_id = $_;

  opendir my $dh, "$data_base/$dataset/wav/$spkr_id/" or die "Cannot open directory: $!";
  my @rec_dirs = grep {-d "$data_base/$dataset/wav/$spkr_id/$_" && ! /^\.{1,2}$/} readdir($dh);
  closedir $dh;

  foreach (@rec_dirs) {
    my $rec_id = $_;

    opendir my $dh, "$data_base/$dataset/wav/$spkr_id/$rec_id/" or die "Cannot open directory: $!";
    my @files = map{s/\.[^.]+$//;$_}grep {/\.wav$/} readdir($dh);
    closedir $dh;

    foreach (@files) {
      my $name = $_;
      if (substr($name, 0, 1) ne "."){
        my $wav = "$data_base/$dataset/wav/$spkr_id/$rec_id/$name.wav";
        my $utt_id = "$spkr_id-$rec_id-$name";
        print WAV "$utt_id", " $wav", "\n";
        print SPKR "$utt_id", " $spkr_id", "\n";
      }
    }
  }
}
close(SPKR) or die;
close(WAV) or die;

if (system(
  "utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
system("env LC_COLLATE=C fix_data_dir.sh $out_dir");
if (system("env LC_COLLATE=C validate_data_dir.sh --no-text --no-feats $out_dir") != 0) {
  die "Error validating directory $out_dir";
}
