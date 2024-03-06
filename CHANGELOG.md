# Changelog

## [0.3.0](https://github.com/namphuongtran9196/3m-ser-private/compare/v0.2.1...v0.3.0) (2023-11-07)


### Features

* add combined margin loss, focal loss ([9d71b4e](https://github.com/namphuongtran9196/3m-ser-private/commit/9d71b4e9f388ca4a9f59e9558475b15260ae7dbc))
* add confusion_maxtrix notbook ([eafcd09](https://github.com/namphuongtran9196/3m-ser-private/commit/eafcd092f9740916a4409f583e9b9b517c4ef91e))
* add f1_score, precision, recall ([6ad6f89](https://github.com/namphuongtran9196/3m-ser-private/commit/6ad6f89868c9e15e1b6366c0a76c18e6a287ba62))
* add lstm model ([bcbb629](https://github.com/namphuongtran9196/3m-ser-private/commit/bcbb629da179fe30079039846bb7fd364c0bfc87))
* add MELD dataset, merge v2 scripts ([e559ffc](https://github.com/namphuongtran9196/3m-ser-private/commit/e559ffca7c236af38008ed403f5be9a0e3db9129))
* add opt_path for config ([ac15e8d](https://github.com/namphuongtran9196/3m-ser-private/commit/ac15e8d14cb92ac7f4bf4fe996f98d68609eed9b))
* add resume training ([7fe7150](https://github.com/namphuongtran9196/3m-ser-private/commit/7fe71507f882c5aec6f6ef4e853a00c28907578a))
* add SERVER model, train optimize, ICTC ([6afe441](https://github.com/namphuongtran9196/3m-ser-private/commit/6afe441894ddb7b018f93c7c6732003eebf56a20))
* add test scripts ([04ccaaf](https://github.com/namphuongtran9196/3m-ser-private/commit/04ccaafd1c3a365780140cab969d6bc1a2d80d9f))
* **config:** add load opt from file ([c5eef15](https://github.com/namphuongtran9196/3m-ser-private/commit/c5eef15daf8403f0689bec29b942188f6e46318a))
* fix small sample error, add torchvggish ([2db3d10](https://github.com/namphuongtran9196/3m-ser-private/commit/2db3d105b8dfc6e7ff9c7a0a11fa6d87d1370301))
* lstm, iemocap audio data + preprocessing ([31c4ef2](https://github.com/namphuongtran9196/3m-ser-private/commit/31c4ef27f9ee841e62e7ec4121575896792bfffa))
* model version 2 for adding more config ([1218a92](https://github.com/namphuongtran9196/3m-ser-private/commit/1218a9226a6d28cbd036f2d3b08fe696c6f403f3))
* optional to use learning scheduler ([ea9d721](https://github.com/namphuongtran9196/3m-ser-private/commit/ea9d7216b002bd91ec53b7b3834e9239228c4f35))
* **trainer:** add resume checkpoint ([4206d03](https://github.com/namphuongtran9196/3m-ser-private/commit/4206d036e2d6907f90a8815fb69b642964002d9a))
* **trainer:** add save all states for margin loss ([748c807](https://github.com/namphuongtran9196/3m-ser-private/commit/748c80771d3b8565071033f2aedc4f2e9541924b))


### Bug Fixes

* addition epoch in trainer ([4fd8692](https://github.com/namphuongtran9196/3m-ser-private/commit/4fd86920be0b739bf143e5b4197cf6a895acf1e7))
* load config ([fd6aca9](https://github.com/namphuongtran9196/3m-ser-private/commit/fd6aca94923966a549082f3e54335683067943db))
* **network:** change linear unit in audio model ([fecf7c3](https://github.com/namphuongtran9196/3m-ser-private/commit/fecf7c3a645aa7c214524a2a736aeb065703b31b))
* v2 model ([9e00469](https://github.com/namphuongtran9196/3m-ser-private/commit/9e004692e895d117195e03adb0cfb8f8657cbbd7))
* vggish, 3m-ser_v2 ([51804d0](https://github.com/namphuongtran9196/3m-ser-private/commit/51804d03ce01c478bc3ceabbc44003fed3dde018))


### Performance Improvements

* merge train v2 model scripts ([a12f629](https://github.com/namphuongtran9196/3m-ser-private/commit/a12f629921e9d90282110f5df347e850ee40dd3a))


### Documentation

* update citation ([e85c791](https://github.com/namphuongtran9196/3m-ser-private/commit/e85c791c43a5e82e6c00b55659e093ac842de0fe))

## [0.2.1](https://github.com/namphuongtran9196/3m-ser-private/compare/v0.2.0...v0.2.1) (2023-07-16)


### Documentation

* update train scripts, add pretrained link ([428c95e](https://github.com/namphuongtran9196/3m-ser-private/commit/428c95e11011236b735d455126caf74e96398c89))

## [0.2.0](https://github.com/namphuongtran9196/3m-ser-private/compare/v0.1.0...v0.2.0) (2023-07-16)


### Features

* add audio augmentation for panns ([3c25ad9](https://github.com/namphuongtran9196/3m-ser-private/commit/3c25ad932341d50390447a3e57e7476b9bb54207))
* add hubert, wav2vec, wavlm ([4b3ae3e](https://github.com/namphuongtran9196/3m-ser-private/commit/4b3ae3ef054d6dc84d9418c30a4a21538aba3298))
* add model for audio only ([5ebabb6](https://github.com/namphuongtran9196/3m-ser-private/commit/5ebabb674022270779b74d413db6b210b841905f))
* add panns model ([83b350d](https://github.com/namphuongtran9196/3m-ser-private/commit/83b350d30e9f984daf40c787c0696cedb70e0a9a))
* add PANNs model for audio ([3e87528](https://github.com/namphuongtran9196/3m-ser-private/commit/3e8752838c23fe90656596e7ac9839792b3fcab5))
* add roberta for text encoder ([45868aa](https://github.com/namphuongtran9196/3m-ser-private/commit/45868aa53795f28259231dcf42d369cc209716ca))
* add simple fusion network ([131856a](https://github.com/namphuongtran9196/3m-ser-private/commit/131856a0c969ace35f3f0101170fd57df5ee99c4))
* add TextOnly and AudioOnly model ([8975746](https://github.com/namphuongtran9196/3m-ser-private/commit/897574680985a2857b26645c5aa2ef0ba9165590))
* batch training, change panns to token emb ([4d89099](https://github.com/namphuongtran9196/3m-ser-private/commit/4d89099baa776266aee1dd82f32886ab8c0a9c21))
* merge all model into one scripts ([cf7a129](https://github.com/namphuongtran9196/3m-ser-private/commit/cf7a129333a93ab53833c5928657e6679604db0a))
* train script, config for panns ([cc2dcfc](https://github.com/namphuongtran9196/3m-ser-private/commit/cc2dcfce0b1df5d4292547bc39f84db8e2c982b8))


### Bug Fixes

* 3m-ser module with batch size 1 ([5ae5fc0](https://github.com/namphuongtran9196/3m-ser-private/commit/5ae5fc0e1d250820f1f31b3595097b08b3b3e0ec))
* load weights, dataloader ([a14f211](https://github.com/namphuongtran9196/3m-ser-private/commit/a14f2111a01878f2f3fd78f305d1f3a6efa5b506))
* model name import error ([6ac64fb](https://github.com/namphuongtran9196/3m-ser-private/commit/6ac64fb783a67704f4bf55a1d6d4b9039123d21a))
* **modules:** import Wavegram_Logmel_Cnn14 ([b35e1fb](https://github.com/namphuongtran9196/3m-ser-private/commit/b35e1fb17cece6559351a6b07dbd8e38c55a4b09))
* **network:** fc dimension for audio model ([bf9443e](https://github.com/namphuongtran9196/3m-ser-private/commit/bf9443edb924aee2a60ec95be63613cc2a234899))
* preprocess data for audio ([34512e9](https://github.com/namphuongtran9196/3m-ser-private/commit/34512e9f681c4081a5b5f0699845e20417f03523))


### Documentation

* add h5py dependence ([147e717](https://github.com/namphuongtran9196/3m-ser-private/commit/147e7174fcc8cb1772917bd957b0a14cfb5bbd20))
* update citation ([059b0b5](https://github.com/namphuongtran9196/3m-ser-private/commit/059b0b5bd0714691cecb0daa012bb7096385ae5e))
* update citation ([f5f077b](https://github.com/namphuongtran9196/3m-ser-private/commit/f5f077ba5286c95634dedf291596639ce8d3fc02))
* update citation and acc readme ([778dfa7](https://github.com/namphuongtran9196/3m-ser-private/commit/778dfa76ec13930863466fa848722331c26db2d3))
* update citation readme ([edf03d6](https://github.com/namphuongtran9196/3m-ser-private/commit/edf03d62fbe7d85775088585dee30046179b2f36))
* update requirements for panns model ([e61eea2](https://github.com/namphuongtran9196/3m-ser-private/commit/e61eea23dfde2d452b0c8fc7b4b745e4be503c94))
* upgrade to pytorch2.0.1 ([ee12cba](https://github.com/namphuongtran9196/3m-ser-private/commit/ee12cba396e89b18b08e27228e71a0f4536b2d24))

## 0.1.0 (2023-06-08)


### Features

* add 3m-ser model ([0b89766](https://github.com/namphuongtran9196/3m-ser-private/commit/0b897667065615a788853ff8c4d5f5cfd2ac5a58))
* add attention visualize ([29a040f](https://github.com/namphuongtran9196/3m-ser-private/commit/29a040f1c4b2a32410433c6724a0cd786cde2fe7))
* add preprocess scripts ([ca120e8](https://github.com/namphuongtran9196/3m-ser-private/commit/ca120e84a5e89e27de809565762d5277ca2a3a59))
* add visualization modules ([5e9ba03](https://github.com/namphuongtran9196/3m-ser-private/commit/5e9ba030d21e6545062ca166aa142c7c3969d291))


### Documentation

* add citation, update instruction ([3e869a4](https://github.com/namphuongtran9196/3m-ser-private/commit/3e869a4784be3394c2c13b617c7af1e86e9eda88))
* add license, update readme.md ([171fc43](https://github.com/namphuongtran9196/3m-ser-private/commit/171fc4378c1cd988b28bccc3e6c195ccf166e5ac))
* update readme ([7918781](https://github.com/namphuongtran9196/3m-ser-private/commit/7918781ac5cd46068a24985a966f61a1f3eff59d))
