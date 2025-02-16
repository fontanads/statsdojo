# Practicing with Stan PPL

- [Stan's official documentation](https://mc-stan.org/docs/)

- [Download and install guides](https://mc-stan.org/docs/)

- [`cmdstanpy` Python inteface](https://mc-stan.org/cmdstanpy/index.html#module-cmdstanpy)

## My Experience installing Stan on Windows and using cmdstanpy
  - In the install page, I've chosen OS Windows with CmdStan interface
  - I cloned the repo `git clone https://github.com/stan-dev/cmdstan.git --recursive` in my "MyUserHome" user folder
  - I used a MinGW64 shell, i.e. **git bash** from [Git for Windows](https://gitforwindows.org/), opening a new tab in PowerShell (it can be done in VS Code Terminal as well); once you have the MINGW64 shell, check `where g++` and `where make` to be sure it has the necessary requirements.
  - I ran `make build` on the cloned repo `myhome/cmdstan/`
  - I set the `Path` of Windows User Environment Variables to include `PathToMyUserHome\cmdstan\stan\lib\stan_math\lib\tbb`
  - I used poetry to add `cmdstanpy` (or just pip install it in your local Python venv)
  - In my Python code, you'll see me doing the following:
  ```
  # set the path to the CmdStan installation
  from cmdstanpy import cmdstan_path, set_cmdstan_path
  set_cmdstan_path(os.path.join("C:", "Users", "MyUserHome", "cmdstan"))
  cmdstan_path()
  ```

That should get you all set in Windows OS.

## Practice folder

I'm practing on [./stan/](./stan/) in this repo.  
More to come soon.  
