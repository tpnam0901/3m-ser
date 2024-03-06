# Changelog

## [1.1.0](https://github.com/namphuongtran9196/3m-ser/compare/v1.0.0...v1.1.0) (2024-03-06)


### Features

* add cel+ccl for audio and text, format code ([96fac67](https://github.com/namphuongtran9196/3m-ser/commit/96fac679d22b06f7400842b16ee6794397a957a5))
* add confusion matrix ([5811b81](https://github.com/namphuongtran9196/3m-ser/commit/5811b81d18b8357564353023d8dcee8b2329f7e3))
* add new module, improve performance ([00275a0](https://github.com/namphuongtran9196/3m-ser/commit/00275a05259c6fa41493bc2a7e5be87c28e3cd63))
* add optim loss, encode data, esd config ([d219d10](https://github.com/namphuongtran9196/3m-ser/commit/d219d10d0fb96ff8b8ea5501791a3629b438bb65))
* add plit dataset notebook ([dfbb67e](https://github.com/namphuongtran9196/3m-ser/commit/dfbb67ec8ea2dc93ddf3bf8bc74fa9cd857c8b89))
* add read mlflow notebook ([fa5e670](https://github.com/namphuongtran9196/3m-ser/commit/fa5e670b21739e596232b6f223a128448fdde402))
* add step on mlflow logging ([895d8d8](https://github.com/namphuongtran9196/3m-ser/commit/895d8d856ac074f0ff790929225d4486c2652986))
* cosine scores plot, pca2D ([1b422a8](https://github.com/namphuongtran9196/3m-ser/commit/1b422a8bff980d1da42870bfba6b710ae07ad575))


### Bug Fixes

* eval single model, save weight callback ([a7818bb](https://github.com/namphuongtran9196/3m-ser/commit/a7818bb43648e9e8b679b2f6f7512e707996ae16))
* update script to current version ([77a1259](https://github.com/namphuongtran9196/3m-ser/commit/77a1259e6c862c5616dea43b71b3b97eab1fa110))

## [1.0.0](https://github.com/namphuongtran9196/3m-ser/compare/v0.2.0...v1.0.0) (2023-11-07)


### ⚠ BREAKING CHANGES

* upgrade to version 1.0

### Features

* add audio augmentation for panns ([3c25ad9](https://github.com/namphuongtran9196/3m-ser/commit/3c25ad932341d50390447a3e57e7476b9bb54207))
* add combined margin loss, focal loss ([9d71b4e](https://github.com/namphuongtran9196/3m-ser/commit/9d71b4e9f388ca4a9f59e9558475b15260ae7dbc))
* add confusion_maxtrix notbook ([eafcd09](https://github.com/namphuongtran9196/3m-ser/commit/eafcd092f9740916a4409f583e9b9b517c4ef91e))
* add f1_score, precision, recall ([6ad6f89](https://github.com/namphuongtran9196/3m-ser/commit/6ad6f89868c9e15e1b6366c0a76c18e6a287ba62))
* add hubert, wav2vec, wavlm ([4b3ae3e](https://github.com/namphuongtran9196/3m-ser/commit/4b3ae3ef054d6dc84d9418c30a4a21538aba3298))
* add lstm model ([bcbb629](https://github.com/namphuongtran9196/3m-ser/commit/bcbb629da179fe30079039846bb7fd364c0bfc87))
* add MELD dataset, merge v2 scripts ([e559ffc](https://github.com/namphuongtran9196/3m-ser/commit/e559ffca7c236af38008ed403f5be9a0e3db9129))
* add model for audio only ([5ebabb6](https://github.com/namphuongtran9196/3m-ser/commit/5ebabb674022270779b74d413db6b210b841905f))
* add opt_path for config ([ac15e8d](https://github.com/namphuongtran9196/3m-ser/commit/ac15e8d14cb92ac7f4bf4fe996f98d68609eed9b))
* add panns model ([83b350d](https://github.com/namphuongtran9196/3m-ser/commit/83b350d30e9f984daf40c787c0696cedb70e0a9a))
* add PANNs model for audio ([3e87528](https://github.com/namphuongtran9196/3m-ser/commit/3e8752838c23fe90656596e7ac9839792b3fcab5))
* add resume training ([7fe7150](https://github.com/namphuongtran9196/3m-ser/commit/7fe71507f882c5aec6f6ef4e853a00c28907578a))
* add roberta for text encoder ([45868aa](https://github.com/namphuongtran9196/3m-ser/commit/45868aa53795f28259231dcf42d369cc209716ca))
* add SERVER model, train optimize, ICTC ([6afe441](https://github.com/namphuongtran9196/3m-ser/commit/6afe441894ddb7b018f93c7c6732003eebf56a20))
* add simple fusion network ([131856a](https://github.com/namphuongtran9196/3m-ser/commit/131856a0c969ace35f3f0101170fd57df5ee99c4))
* add test scripts ([04ccaaf](https://github.com/namphuongtran9196/3m-ser/commit/04ccaafd1c3a365780140cab969d6bc1a2d80d9f))
* add TextOnly and AudioOnly model ([8975746](https://github.com/namphuongtran9196/3m-ser/commit/897574680985a2857b26645c5aa2ef0ba9165590))
* batch training, change panns to token emb ([4d89099](https://github.com/namphuongtran9196/3m-ser/commit/4d89099baa776266aee1dd82f32886ab8c0a9c21))
* **config:** add load opt from file ([c5eef15](https://github.com/namphuongtran9196/3m-ser/commit/c5eef15daf8403f0689bec29b942188f6e46318a))
* fix small sample error, add torchvggish ([2db3d10](https://github.com/namphuongtran9196/3m-ser/commit/2db3d105b8dfc6e7ff9c7a0a11fa6d87d1370301))
* lstm, iemocap audio data + preprocessing ([31c4ef2](https://github.com/namphuongtran9196/3m-ser/commit/31c4ef27f9ee841e62e7ec4121575896792bfffa))
* merge all model into one scripts ([cf7a129](https://github.com/namphuongtran9196/3m-ser/commit/cf7a129333a93ab53833c5928657e6679604db0a))
* model version 2 for adding more config ([1218a92](https://github.com/namphuongtran9196/3m-ser/commit/1218a9226a6d28cbd036f2d3b08fe696c6f403f3))
* optional to use learning scheduler ([ea9d721](https://github.com/namphuongtran9196/3m-ser/commit/ea9d7216b002bd91ec53b7b3834e9239228c4f35))
* train script, config for panns ([cc2dcfc](https://github.com/namphuongtran9196/3m-ser/commit/cc2dcfce0b1df5d4292547bc39f84db8e2c982b8))
* **trainer:** add resume checkpoint ([4206d03](https://github.com/namphuongtran9196/3m-ser/commit/4206d036e2d6907f90a8815fb69b642964002d9a))
* **trainer:** add save all states for margin loss ([748c807](https://github.com/namphuongtran9196/3m-ser/commit/748c80771d3b8565071033f2aedc4f2e9541924b))


### Bug Fixes

* 3m-ser module with batch size 1 ([5ae5fc0](https://github.com/namphuongtran9196/3m-ser/commit/5ae5fc0e1d250820f1f31b3595097b08b3b3e0ec))
* addition epoch in trainer ([4fd8692](https://github.com/namphuongtran9196/3m-ser/commit/4fd86920be0b739bf143e5b4197cf6a895acf1e7))
* load config ([fd6aca9](https://github.com/namphuongtran9196/3m-ser/commit/fd6aca94923966a549082f3e54335683067943db))
* load weights, dataloader ([a14f211](https://github.com/namphuongtran9196/3m-ser/commit/a14f2111a01878f2f3fd78f305d1f3a6efa5b506))
* model name import error ([6ac64fb](https://github.com/namphuongtran9196/3m-ser/commit/6ac64fb783a67704f4bf55a1d6d4b9039123d21a))
* **modules:** import Wavegram_Logmel_Cnn14 ([b35e1fb](https://github.com/namphuongtran9196/3m-ser/commit/b35e1fb17cece6559351a6b07dbd8e38c55a4b09))
* **network:** change linear unit in audio model ([fecf7c3](https://github.com/namphuongtran9196/3m-ser/commit/fecf7c3a645aa7c214524a2a736aeb065703b31b))
* **network:** fc dimension for audio model ([bf9443e](https://github.com/namphuongtran9196/3m-ser/commit/bf9443edb924aee2a60ec95be63613cc2a234899))
* preprocess data for audio ([34512e9](https://github.com/namphuongtran9196/3m-ser/commit/34512e9f681c4081a5b5f0699845e20417f03523))
* v2 model ([9e00469](https://github.com/namphuongtran9196/3m-ser/commit/9e004692e895d117195e03adb0cfb8f8657cbbd7))
* vggish, 3m-ser_v2 ([51804d0](https://github.com/namphuongtran9196/3m-ser/commit/51804d03ce01c478bc3ceabbc44003fed3dde018))


### Performance Improvements

* merge train v2 model scripts ([a12f629](https://github.com/namphuongtran9196/3m-ser/commit/a12f629921e9d90282110f5df347e850ee40dd3a))


### Documentation

* add h5py dependence ([147e717](https://github.com/namphuongtran9196/3m-ser/commit/147e7174fcc8cb1772917bd957b0a14cfb5bbd20))
* update citation ([e85c791](https://github.com/namphuongtran9196/3m-ser/commit/e85c791c43a5e82e6c00b55659e093ac842de0fe))
* update citation ([059b0b5](https://github.com/namphuongtran9196/3m-ser/commit/059b0b5bd0714691cecb0daa012bb7096385ae5e))
* update citation ([f5f077b](https://github.com/namphuongtran9196/3m-ser/commit/f5f077ba5286c95634dedf291596639ce8d3fc02))
* update citation and acc readme ([778dfa7](https://github.com/namphuongtran9196/3m-ser/commit/778dfa76ec13930863466fa848722331c26db2d3))
* update citation readme ([edf03d6](https://github.com/namphuongtran9196/3m-ser/commit/edf03d62fbe7d85775088585dee30046179b2f36))
* update requirements for panns model ([e61eea2](https://github.com/namphuongtran9196/3m-ser/commit/e61eea23dfde2d452b0c8fc7b4b745e4be503c94))
* update train scripts, add pretrained link ([428c95e](https://github.com/namphuongtran9196/3m-ser/commit/428c95e11011236b735d455126caf74e96398c89))
* upgrade to pytorch2.0.1 ([ee12cba](https://github.com/namphuongtran9196/3m-ser/commit/ee12cba396e89b18b08e27228e71a0f4536b2d24))

## 0.1.0 (2023-06-09)


### Features

* add 3m-ser model ([df0c507](https://github.com/namphuongtran9196/3m-ser/commit/df0c5076a1a71df6a80ff123cd9687ed6da47023))
* add version v0.1.0 ([9ea3762](https://github.com/namphuongtran9196/3m-ser/commit/9ea376289c32cbe8fbb3cbd258fff9aeb4bb3d36))


### Documentation

* add abstract ([21ca06d](https://github.com/namphuongtran9196/3m-ser/commit/21ca06ded5bf5067ba65c2dbfdcf6389485756ec))
* add citation, references section ([e68f5ff](https://github.com/namphuongtran9196/3m-ser/commit/e68f5ff4a11327a2085bed6984f8e573321c7c07))
* add LICENSE ([a3ca361](https://github.com/namphuongtran9196/3m-ser/commit/a3ca3614ca21ab84d92ae30e747bcf435890924a))
* draft readme.md ([ee6e568](https://github.com/namphuongtran9196/3m-ser/commit/ee6e568dd8c412d61b2565686bf794ebddc03062))
* refactor title ([3b05efb](https://github.com/namphuongtran9196/3m-ser/commit/3b05efbe80d0a2ee57a8af6d93763200ca113765))
* update readme, repo settings ([7c47bf6](https://github.com/namphuongtran9196/3m-ser/commit/7c47bf6ba734d4b9d03bfbf5932821dc0fbfe166))
