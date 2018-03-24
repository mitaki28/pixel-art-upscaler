const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin')
const LicenseWebpackPlugin = require('license-webpack-plugin').LicenseWebpackPlugin;

module.exports = {
  mode: "development",
  entry: ["loaders.css", './src/index.tsx'],
  devtool: 'inline-source-map',
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/
      },
      {
        test: /\.css?$/,
        use: [
          { loader: "style-loader" },
          { loader: "css-loader" }
        ]
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
    }),
    new LicenseWebpackPlugin({
      pattern: /.*/,
    }),
  ],
  node: {
    fs: 'empty'
  }
};