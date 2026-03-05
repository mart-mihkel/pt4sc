{
  pkgs ? import <nixpkgs> {},
  run ? "zsh",
}: let
  fhs = pkgs.buildFHSEnv {
    name = "fhs-shell";
    targetPkgs = pkgs: with pkgs; [uv zlib];
    runScript = "${run}";
  };
in
  fhs.env
