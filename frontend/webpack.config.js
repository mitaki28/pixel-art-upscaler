const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin')

module.exports = {
  mode: "development",
  entry: './src/index.tsx',
  devtool: 'inline-source-map',
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/
      }
    ]
  },
  resolve: {
    extensions: [ '.tsx', '.ts', '.js' ]
  },
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'public')
  },
  externals: {
    "jimp": "Jimp"
  },
  devServer: {
    contentBase: path.join(__dirname, "public"),
    port: 9000
  },
  plugins: [
    new HtmlWebpackPlugin({
      hash: true,
      template: path.join(__dirname, "src/index.html"),
    })
  ],
  node: {
    fs: 'empty'
  }
};