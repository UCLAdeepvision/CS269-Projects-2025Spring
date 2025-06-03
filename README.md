# UCLA CS 269 Course Project

**Project page: https://ucladeepvision.github.io/CS269-Projects-2025Spring/**


## Instruction for running this site locally

1. Follow the first 2 steps in [pull-request-instruction](pull-request-instruction.md)

2. Installing Ruby with version 3.1.4 

For MacOS:
```
brew install rbenv ruby-build
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
rbenv install 3.1.4 && rbenv global 3.1.4
```
For Ubuntu: 
```
curl -fsSL https://github.com/rbenv/rbenv-installer/raw/HEAD/bin/rbenv-installer | bash
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc
rbenv install 3.1.4 && rbenv global 3.1.4
```

Check your Ruby version
```
ruby -v # should be 3.1.4
```

3. Installing Bundler and jekyll with
```
gem install --user-install bundler jekyll
bundler install
```

4. Run your site with
```
bundle exec jekyll serve
```
You should see an address pop on the terminal (http://127.0.0.1:4000/CS269-Projects-2025Spring/ by default), go to this address with your browser.

## Working on the project

1. Create a folder with your team id under ```./assets/images/student-id```. For example: ```./assets/images/student-01```. You will use this folder to store all the images in your project. Please check the column E of this sheet https://docs.google.com/spreadsheets/d/1TmYbvt9rT6PmRR8t9ZsbCaW9MFj644dfHYDSLGFXtpI/edit?usp=sharing for your id number.

2. Copy the template at "2024-12-13-student-01-peekaboo.md" and rename it with format "yyyy-mm-dd-student-XX-projectshortname.md" under ```./_posts/```.

3. Check out the sample post we provide at https://ucladeepvision.github.io/CS269-Projects-2025Spring/ and the source code at https://raw.githubusercontent.com/UCLAdeepvision/CS269-Projects-2025Spring/main/_posts/2024-12-13-student-01-peekaboo.md as well as basic Markdown syntax at https://www.markdownguide.org/basic-syntax/

4. Start your work in your .md file. You may **only** edit the .md file you just copied and renamed, and add images to ```./assets/images/student-id```. *Please do NOT change any other files in this repo.*

Once you save the .md file, jekyll will synchronize the site and you can check the changes on browser.

## Submission
We will use git pull request to manage submissions.

Once you've done, follow steps 3 and 4 in [pull-request-instruction](pull-request-instruction.md) to make a pull request BEFORE the deadline. Please make sure not to modify any file except your .md file and your images folder. We will merge the request after all submissions are received, and you should able to check your work in the project page on next week of each deadline.

## Deadlines  
Please update your final blog post by submitting a pull request by **11:59 PM on Tuesday of Week 11 (June 10)**.

-----

Kudos to [Tianpei](https://gutianpei.github.io/), who originally developed this site for CS 188 in Winter 2022.
